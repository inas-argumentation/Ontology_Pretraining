import itertools
import json
import os
from load_models_and_data.load_raw_biology_data import BioDataset
from torch.utils.data import Dataset
import numpy as np
import random
from collections import Counter
from nltk.stem import PorterStemmer
from settings import data_dir
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
import torch
from tqdm import tqdm

def load_filenames(split):
    with open(data_dir("datasets/abstract_splits.json"), "r") as f:
        return json.load(f)[split]

class BaseSIMDataset(Dataset):
    def __init__(self, split="5000", seed=42, use_relatedness_loss=True):
        self.seed = seed
        self.split = split
        self.use_relatedness_loss = use_relatedness_loss

    def init_dataset(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens_seen = 0
        self.batches_seen = 0

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.train_data = self.load_data()
        self.concept_list = list(self.train_data.keys())
        self.reset_epoch_sampling()

    def load_data(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def reset_epoch_sampling(self):
        if not self.use_relatedness_loss:
            self.current_epoch_concepts = self.concept_list.copy()
            random.shuffle(self.current_epoch_concepts)
            return

        n_concepts = len(self.concept_list)
        # Pre-compute similar pairs for each concept
        similar_pairs_dict = {}
        similarities = self.concept_similarities.tocsr()  # Convert to CSR once for efficient row slicing

        result = []
        available_indices = np.arange(n_concepts)
        np.random.shuffle(available_indices)

        # Start with first concept
        current_idx = available_indices[0]
        result.append(self.concept_list[current_idx])
        available_indices = available_indices[1:]

        while len(available_indices) > 0:
            # Get or compute similar indices for current concept
            if current_idx not in similar_pairs_dict:
                row = similarities[current_idx].toarray().ravel()
                similar_pairs_dict[current_idx] = np.where(row > 0)[0]

            # Find intersection of similar indices and available indices
            similar_indices = np.intersect1d(similar_pairs_dict[current_idx], available_indices)

            # Choose next index
            if len(similar_indices) > 0 and np.random.random() < 0.9:
                next_idx = np.random.choice(similar_indices)
            else:
                next_idx = np.random.choice(available_indices)

            result.append(self.concept_list[next_idx])
            available_indices = available_indices[available_indices != next_idx]
            current_idx = next_idx

            # Add one more random concept if available
            if len(available_indices) > 0:
                next_idx = np.random.choice(available_indices)
                result.append(self.concept_list[next_idx])
                available_indices = available_indices[available_indices != next_idx]
                current_idx = next_idx

        self.current_epoch_concepts = result


    def sample_batch_concepts(self, batch_size):
        if len(self.current_epoch_concepts) < batch_size:
            # Start a new epoch if not enough concepts left
            remaining_concepts = self.current_epoch_concepts.copy()
            self.reset_epoch_sampling()
            needed = batch_size - len(remaining_concepts)
            batch_concepts = remaining_concepts + self.current_epoch_concepts[:needed]
            self.current_epoch_concepts = self.current_epoch_concepts[needed:]
        else:
            batch_concepts = self.current_epoch_concepts[:batch_size]
            self.current_epoch_concepts = self.current_epoch_concepts[batch_size:]
        return batch_concepts

    def get_batch(self, batch_size):
        concepts = self.sample_batch_concepts(batch_size)
        sentences = []

        for concept in concepts:
            definitions = random.choice(self.train_data[concept])
            idx1, idx2 = random.sample(range(len(definitions)), 2)
            sentences.extend([definitions[idx1], definitions[idx2]])

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        self.tokens_seen += sum(len(self.tokenizer.tokenize(s)[:512]) for s in sentences)

        if self.use_relatedness_loss:
            similarity_matrix = np.zeros((batch_size, batch_size))
            for idx_1, concept_1 in enumerate(concepts):
                for idx_2, concept_2 in enumerate(concepts):
                    similarity_matrix[idx_1, idx_2] = 1 if self.concept_similarities[self.concept_indices[concept_1], self.concept_indices[concept_2]] else 0

            return inputs, similarity_matrix

        return inputs, None

    def get_sentence_list(self):
        sentences = []
        for concept in self.train_data:
            for sentence in self.train_data[concept]:
                sentences.append(f"{concept}: {sentence}")
        return sentences


class OntologyDefinitionSIMDataset(BaseSIMDataset):
    dataset_name = "onto_definitions"

    def load_data(self):
        with open(data_dir("processed_datasets/ontology_definition_dataset.json"), 'r') as f:
            data = json.load(f)

        concept_dict = {}
        for item in data.values():
            if len(item['definitions']) == 0:
                continue
            concept = item['concept']
            definitions = [item['definitions'][0]] if len(item['definitions']) > 0 else []
            definitions.extend(item['generated_definitions'])
            concept_dict[concept] = [definitions]

        if self.use_relatedness_loss:
            concept_list = list(concept_dict.keys())
            self.concept_indices = {x: idx for idx, x in enumerate(concept_list)}
            self.concept_similarities = np.zeros((len(concept_list), len(concept_list)), dtype="int")

            for item in data.values():
                if len(item['definitions']) < 1 or len(item['definitions'][0]) < 2:
                    continue

                concept = item['concept']
                for parent in item["parents"]:
                    if parent in self.concept_indices:
                        self.concept_similarities[self.concept_indices[concept]][self.concept_indices[parent]] = 1
                        self.concept_similarities[self.concept_indices[parent]][self.concept_indices[concept]] = 1
                for connection in item["connections"]:
                    if connection in self.concept_indices:
                        self.concept_similarities[self.concept_indices[concept]][self.concept_indices[connection]] = 1
                        self.concept_similarities[self.concept_indices[connection]][self.concept_indices[concept]] = 1
            self.concept_similarities = csr_matrix(self.concept_similarities)
        return concept_dict

class AbstractKeywordDefinitionSIMDataset(BaseSIMDataset):
    dataset_name = "abstract_definitions"

    ps = PorterStemmer()

    def normalize_concept_name(self, concept):
        concept = concept.replace("-", " ")
        words = concept.split(" ")
        stemmed = [self.ps.stem(w) for w in words]
        return " ".join(stemmed)

    def compute_document_vectors(self, filenames):
        num_concepts = len(self.concept_indices)
        num_docs = len(filenames)

        row_indices = []
        col_indices = []
        values = []

        concept_to_row = {concept: idx for idx, concept in enumerate(self.concept_indices.keys())}

        for doc_idx, filename in enumerate(tqdm(filenames, desc="Computing document vectors")):
            with open(data_dir(f"processed_datasets/unlabeled_keywords/{int(filename)}.json"), "r") as f:
                data = json.load(f)
                for key in data:
                    key_code = self.normalize_concept_name(key)
                    if key_code in self.concept_indices:
                        row_indices.append(concept_to_row[key_code])
                        col_indices.append(doc_idx)
                        values.append(1)

        concept_doc_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_concepts, num_docs))

        return concept_doc_matrix

    def load_data(self):
        filenames = load_filenames(self.split)

        concepts = {}
        for filename in filenames:
            with open(data_dir(f"processed_datasets/unlabeled_definitions/{int(filename)}.json"), "r") as f:
                data = json.load(f)
                for key in data:
                    key_code = self.normalize_concept_name(key)
                    if len(data[key]) >= 2:
                        if key_code in concepts:
                            concepts[key_code]["definitions"].append(data[key])
                            concepts[key_code]["concept_names"].add(key)
                        else:
                            concepts[key_code] = {"concept_names": {key}, "definitions": [data[key]]}

        new_data_dict = {}
        for key in concepts:
            counter = Counter(concepts[key]["concept_names"])
            most_common_element, count = counter.most_common(1)[0]
            new_data_dict[most_common_element] = concepts[key]["definitions"]

        if self.use_relatedness_loss:
            concept_indices = {x: idx for idx, x in enumerate(list(concepts.keys()))}
            self.concept_similarities = lil_matrix((len(concept_indices), len(concept_indices)), dtype="int")
            concept_counts = defaultdict(int)

            for filename in tqdm(filenames):
                with open(data_dir(f"processed_datasets/unlabeled_keywords/{int(filename)}.json"), "r") as f:
                    data = json.load(f)
                    indices = []
                    for key in data:
                        key_code = self.normalize_concept_name(key)
                        if key_code in concept_indices:
                            indices.append(concept_indices[key_code])
                            concept_counts[key] += 1
                    indices = list(set(indices))
                    for x in indices:
                        for y in indices:
                            self.concept_similarities[x, y] += 1

            self.concept_similarities.setdiag(0)
            self.concept_similarities = self.concept_similarities.tocsr()

            threshold = {"5000": 3, "15000": 4, "25000": 5, "35000": 6}[self.split]
            self.concept_similarities.data[self.concept_similarities.data < threshold] = 0
            self.concept_similarities.eliminate_zeros()

            self.concept_indices = concept_indices
            self.concept_doc_matrix = self.compute_document_vectors(filenames)

            self.concept_indices = {}
            for key in concept_indices:
                counter = Counter(concepts[key]["concept_names"])
                most_common_element, count = counter.most_common(1)[0]
                self.concept_indices[most_common_element] = concept_indices[key]

        return new_data_dict