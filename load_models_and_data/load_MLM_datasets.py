import itertools
import json
import torch
from torch.utils.data import Dataset
from blingfire import text_to_sentences
import numpy as np
import random
from nltk.stem import PorterStemmer
from settings import data_dir

def load_filenames(split):
    with open(data_dir("datasets/abstract_splits.json"), "r") as f:
        return json.load(f)[split]

class BaseMLMDataset(Dataset):

    def __init__(self, split="5000", seed=42, max_length=510):
        self.max_length = max_length
        self.seed = seed
        self.split = split

    def init_dataset(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens_seen = 0
        self.batches_seen = 0

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Load dataset-specific data
        self.train_samples = self.load_data()
        self.current_epoch_indices = []
        self.reset_epoch_sampling()

    def load_data(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def reset_epoch_sampling(self):
        self.current_epoch_indices = list(range(len(self.train_samples)))
        random.shuffle(self.current_epoch_indices)

    def _create_mlm_mask(self, tokens, mask_prob=0.25):
        mask = torch.full(tokens.shape, False, dtype=torch.bool)

        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in [tokens.tolist()]
        ]
        mask[torch.tensor(special_tokens_mask[0], dtype=torch.bool)] = False

        # Get positions to mask
        indices = torch.arange(len(tokens))[~mask]
        num_to_mask = max(1, int(round(len(indices) * mask_prob)))
        mask_indices = np.random.choice(indices, num_to_mask, replace=False)

        return mask_indices

    def _count_real_tokens(self, input_ids):
        padding_mask = input_ids != self.tokenizer.pad_token_id
        return padding_mask.sum().item()

    def sample_batch_indices(self, batch_size):
        if len(self.current_epoch_indices) < batch_size:
            # Start a new epoch if not enough indices left
            remaining_indices = self.current_epoch_indices.copy()
            self.reset_epoch_sampling()
            needed = batch_size - len(remaining_indices)
            batch_indices = remaining_indices + self.current_epoch_indices[:needed]
            self.current_epoch_indices = self.current_epoch_indices[needed:]
        else:
            batch_indices = self.current_epoch_indices[:batch_size]
            self.current_epoch_indices = self.current_epoch_indices[batch_size:]
        return batch_indices

    def get_batch(self, batch_size):
        indices = self.sample_batch_indices(batch_size)
        sentences = [self.train_samples[i] for i in indices]

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        masks = [self._create_mlm_mask(seq) for seq in inputs['input_ids']]

        # Update counters
        self.tokens_seen += self._count_real_tokens(inputs['input_ids'])

        return inputs, masks

    def __len__(self):
        return len(self.train_samples)


class AbstractSentenceMLMDataset(BaseMLMDataset):

    dataset_name = "abstract_sentences"

    def __init__(self, max_length=160, **kwargs):
        super(AbstractSentenceMLMDataset, self).__init__(max_length=max_length, **kwargs)

    def load_data(self):
        filenames = load_filenames(self.split)
        sentences = []
        for filename in filenames:
            with open(data_dir(f"datasets/INAS_dataset/unlabeled abstracts/{filename}.txt"), "r") as f:
                text = f.read().strip()
                title, abstract, _ = text.split("\n")
                sentences += [x for x in [title] + text_to_sentences(abstract).split("\n") if len(x) >= 70]

        return sentences

class AbstractMLMDataset(BaseMLMDataset):

    dataset_name = "abstracts"

    def __init__(self, max_length=510, **kwargs):
        super(AbstractMLMDataset, self).__init__(max_length=max_length, **kwargs)

    def load_data(self):
        filenames = load_filenames(self.split)
        abstracts = []
        for filename in filenames:
            with open(data_dir(f"datasets/INAS_dataset/unlabeled abstracts/{filename}.txt"), "r") as f:
                text = f.read().strip()
                title, abstract, _ = text.split("\n")
                if title[-1].isalpha():
                    title = title + "."
                abstracts.append(f"{title} {abstract}")

        return abstracts

class OntologyDefinitionMLMDataset(BaseMLMDataset):

    dataset_name = "onto_definitions"

    def load_data(self):
        with open(data_dir("processed_datasets/ontology_definition_dataset.json"), 'r') as f:
            data = json.load(f)

        original_definitions = []
        generated_definitions = [[] for _ in range(5)]
        for key in data:
            if len(data[key]["definitions"]) > 0:
                original_definitions.append(f"{data[key]['concept']}: {data[key]['definitions'][0]}")

            all_concept_names = data[key]['synonyms'] + [data[key]['concept']]
            for def_idx in range(5):
                if def_idx == len(data[key]["generated_definitions"]):
                    break
                sentence = f"{all_concept_names[def_idx%len(all_concept_names)]}: {data[key]['generated_definitions'][def_idx]}"
                generated_definitions[def_idx].append(sentence)

        for idx in range(len(generated_definitions)):
            random.shuffle(generated_definitions[idx])

        return original_definitions + list(itertools.chain.from_iterable(generated_definitions))


class AbstractKeywordDefinitionMLMDataset(BaseMLMDataset):

    dataset_name = "abstract_definitions"

    def load_data(self):
        ps = PorterStemmer()
        def normalize_concept_name(concept):
            concept = concept.replace("-", " ")
            words = concept.split(" ")
            stemmed = [ps.stem(w) for w in words]
            return " ".join(stemmed)

        filenames = load_filenames(self.split)

        concepts = {}
        for filename in filenames:
            with open(data_dir(f"processed_datasets/unlabeled_definitions/{int(filename)}.json"), "r") as f:
                data = json.load(f)
                for key in data:
                    key_code = normalize_concept_name(key)
                    if key_code in concepts:
                        concepts[key_code]["definitions"] += [(key, d) for d in data[key]]
                        concepts[key_code]["keys"].add(key)
                    else:
                        concepts[key_code] = {"keys": {key}, "definitions": [(key, d) for d in data[key]]}

        for key in concepts:
            random.shuffle(concepts[key]["definitions"])

        all_definitions = []
        for def_idx in range(5):
            for key in concepts:
                if len(concepts[key]["definitions"]) > 0:
                    all_definitions.append(concepts[key]["definitions"].pop())

        final_sentences = [f"{x}: {y}" for x, y in all_definitions]
        return final_sentences


class MixedSentenceAndKeywordDefinitionMLMDataset(BaseMLMDataset):

    dataset_name = "mixed_abstract_sents_and_keyword_defs"

    def load_data(self):
        abstract_sentences = AbstractSentenceMLMDataset(split=self.split).load_data()
        keyword_sentences = AbstractKeywordDefinitionMLMDataset(split=self.split).load_data()
        all_sentences = abstract_sentences + keyword_sentences[:int(len(abstract_sentences)/3)]

        random.shuffle(all_sentences)
        return all_sentences

class MixedSentenceAndOntologyDefinitionMLMDataset(BaseMLMDataset):

    dataset_name = "mixed_abstract_sents_and_keyword_defs"

    def load_data(self):
        abstract_sentences = AbstractSentenceMLMDataset(split=self.split).load_data()
        ontology_sentences = OntologyDefinitionMLMDataset().load_data()
        all_sentences = abstract_sentences + ontology_sentences[-int(len(abstract_sentences)/3):]

        random.shuffle(all_sentences)
        return all_sentences