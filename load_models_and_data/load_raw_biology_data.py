import json
import numpy as np
import os
import sys
from tqdm import tqdm
from settings import data_dir
from load_models_and_data import process_bio_annotations

bio_data_folder = data_dir("datasets/INAS_dataset")

def load_bio_dataset(folder=os.path.join(bio_data_folder, "abstracts_new")):
    result = {}
    files = os.listdir(folder)
    for f in files:
        if f == ".gitkeep":
            continue
        with open(os.path.join(folder, f), "r", encoding="latin-1", newline='') as file:
            title, abstract, sub_labels = file.read().split("\n")
            sub_labels = sub_labels.split(",")
            labels = set([int(x[0]) for x in sub_labels])
            index = int(f[:-4])
            result[index] = {"title": title, "abstract": abstract, "labels": labels, "index": index, "sub_labels": sub_labels}
    return result

def load_bio_unlabeled_abstracts(folder=os.path.join(bio_data_folder, "unlabeled abstracts")):
    result = {}
    files = sorted(os.listdir(folder))
    idx = 0
    for f in tqdm(files, desc="Loading unlabeled abstracts...", file=sys.stdout):
        if f == ".gitkeep":
            continue
        with open(folder + f"/{f}", "r") as file:
            text = file.read().strip()
            title, abstract, doi = text.split("\n")

            index = int(f[:-4])
            result[idx] = {"title": title, "abstract": abstract, "index": index}
            idx += 1
    return result

def load_bio_train_val_test_split_indices():
    test_indices = [int(x) for x in open(os.path.join(bio_data_folder, "test_set_indices.txt"), "r").read().split(",")]
    val_indices = [int(x) for x in open(os.path.join(bio_data_folder, "val_set_indices.txt"), "r").read().split(",")]
    train_indices = [int(x) for x in open(os.path.join(bio_data_folder, "train_set_indices.txt"), "r").read().split(",")]
    return train_indices, val_indices, test_indices

def one_hot_encode(labels):
    vector = np.zeros(10)
    for l in labels:
        vector[l] = 1
    return vector

class BioDataset():
    def __init__(self, skip_unlabeled=False):
        train_indices, val_indices, test_indices = load_bio_train_val_test_split_indices()

        self.indices = {"train": train_indices,
                        "val": val_indices,
                        "test": test_indices}

        self.samples = load_bio_dataset()
        if not skip_unlabeled:
            unlabeled_papers = load_bio_unlabeled_abstracts()
        self.split = "train"

        for k in self.samples.keys():
            self.samples[k]["prediction_text"] = f"{self.samples[k]['title']}. {self.samples[k]['abstract']}"
            self.samples[k]["one_hot_label"] = one_hot_encode(self.samples[k]["labels"])

        if not skip_unlabeled:
            self.unlabeled_papers = {}
            for k in unlabeled_papers.keys():
                index = unlabeled_papers[k]["index"]
                self.unlabeled_papers[index] = unlabeled_papers[k]
                self.unlabeled_papers[index]["prediction_text"] = f"{unlabeled_papers[k]['title']}. {unlabeled_papers[k]['abstract']}"

    def __len__(self):
        return len(self.indices[self.split])

    def __getitem__(self, idx):
        sample = self.samples[self.indices[self.split][idx]]
        return sample["text"], sample["label"]

    def get_full_sample(self, idx):
        return self.samples[idx]

    def set_split(self, split):
        self.split = split

def strip(text):
    text = text.strip()
    if text[0] == ":":
        text = text[1:]
    text = text.strip()
    if text[0] == "\"" or text[0] == "\'":
        text = text[1:]
        if text[-2] == "\"" or text[-2] == "\'":
            text = text[:-2] + text[-1]
    return text

def load_all_keywords():
    labeled_keywords = {}
    unlabeled_keywords = {}
    for folder, keywords_dict in [("processed_datasets/labeled_keywords", labeled_keywords), ("processed_datasets/unlabeled_keywords", unlabeled_keywords)]:
        folder_path = data_dir(folder)
        for filename in tqdm(os.listdir(folder_path), desc="Loading keywords...", file=sys.stdout):
            if filename.endswith('.json'):
                sample_index = int(filename[:-5])
                with open(os.path.join(folder_path, filename), "r") as f:
                    keywords_dict[sample_index] = json.load(f)
    return labeled_keywords, unlabeled_keywords


def load_Bio_annotations():
    with open(os.path.join(bio_data_folder, "annotations.json"), "r") as f:
        annotations = json.loads(f.read())
    return annotations

def create_gt_array_from_annotations(gt_annotations):
    gt_arrays = {}
    for idx, data in gt_annotations.items():
        if len(data) == 0:
            continue
        num_tokens = sum([x["n_tokens"] for x in list(data.values())[0]])
        arrays = []
        for label in range(10):
            if label in data:
                label_arrays = []
                for word in data[label]:
                    if len(word["annotation"]) > 0:
                        for i in range(word["n_tokens"]):
                            label_arrays.append([0, 1])
                    else:
                        for i in range(word["n_tokens"]):
                            label_arrays.append([1, 0])
                arrays.append(np.array(label_arrays))
            else:
                arrays.append(np.array([1, 0]).reshape(1, -1).repeat(num_tokens, 0))
        gt_arrays[idx] = np.stack(arrays, axis=1)
    return gt_arrays

def load_annotations_and_ground_truth(tokenizer, dataset):
    # Load raw annotations from file
    annotations = load_Bio_annotations()
    # Match annotations to tokens created by tokenizer
    annotations = process_bio_annotations.create_gt_annotations(annotations, tokenizer, dataset)

    # Create ground-truth array
    ground_truth_array = create_gt_array_from_annotations(annotations)
    return annotations, ground_truth_array