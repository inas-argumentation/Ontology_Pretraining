import itertools
import math
import os.path
import sys
import warnings

import torch
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import ndcg_score
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"
from load_models_and_data.load_EICAT_data import load_dataset
from settings import data_dir, device
from load_models_and_data.load_models import load_model, save_model

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def create_input(tokenizer, sample, sentence_idx):
    sentences = sample["evidence_sentence_annotation"]
    sentence_tokens = len(tokenizer.tokenize(sentences[sentence_idx][2]))

    add_sentences_before = [x[2] for x in sentences[max(sentence_idx-3, 0):sentence_idx]]
    while len(tokenizer.tokenize(" ".join(add_sentences_before))) > max(1, 490-sentence_tokens):
        add_sentences_before.pop(0)

    input_text = (sample["species"] + f"; " + " ".join(add_sentences_before) + f" {tokenizer.sep_token} " + sentences[sentence_idx][2] +
          f" {tokenizer.sep_token} " + " ".join([x[2] for x in sentences[sentence_idx+1:min(len(sentences), sentence_idx+4)]]))

    return input_text

class EvidenceSentenceDataset_Unbalanced(Dataset):
    def __init__(self, samples, tokenizer):
        self.sentences = []
        self.labels = []

        for sample in samples:
            if sample["evidence_array"] is not None:
                self.sentences += [create_input(tokenizer, sample, i) for i in range(len(sample["evidence_sentence_annotation"]))]
                self.labels += [x[3] for x in sample["evidence_sentence_annotation"]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label': self.labels[idx]
        }


class EvidenceSentenceDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.sentences = []
        self.labels = []

        for sample in samples:
            if sample["evidence_array"] is not None:
                self.sentences += [create_input(tokenizer, sample, i) for i in
                                   range(len(sample["evidence_sentence_annotation"]))]
                self.labels += [x[3] for x in sample["evidence_sentence_annotation"]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label': self.labels[idx]
        }

    def get_class_weights(self, pos_fraction=1/3):
        labels = np.array(self.labels)
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count

        neg_fraction = 1 - pos_fraction

        weights = np.zeros(len(labels), dtype=np.float32)
        if pos_count > 0:
            weights[labels == 1] = pos_fraction / pos_count
        if neg_count > 0:
            weights[labels == 0] = neg_fraction / neg_count

        return torch.FloatTensor(weights)

def collate_fn(batch, tokenizer, max_length=512):
    sentences = [item['sentence'] for item in batch]
    labels = [item['label'] for item in batch]

    inputs = tokenizer(sentences, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to("cuda")
    labels = torch.tensor(labels, dtype=torch.float, device=device)

    return inputs, labels


def evaluate_model(model, tokenizer, dataset, split, metric="nDCG"):
    model.eval()
    nan_to_zero = lambda x: 0 if math.isnan(x) else x
    batch_size = 24
    all_scores = []
    position_scores = []

    with torch.no_grad():
        for sample in tqdm(dataset[split], desc=f"Evaluating {split}", file=sys.stdout):
            if sample["evidence_array"] is not None:
                sentences = sample["evidence_sentence_annotation"]

                texts = [create_input(tokenizer, sample, i) for i in range(len(sentences))]
                labels = [s[3] for s in sentences]

                predictions = []
                for idx in range(0, len(texts), batch_size):
                    batch_inputs = tokenizer(texts[idx:idx+batch_size], return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
                    output = torch.sigmoid(model(**batch_inputs).logits.squeeze(-1)).np()

                    predictions += output.tolist()

                preds_np = np.array(predictions, dtype=float)
                labels_np = np.array(labels, dtype=float)
                binary_predictions = (preds_np > 0.5).astype("float")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    tp = (binary_predictions * labels_np).sum()
                    precision = nan_to_zero(tp / binary_predictions.sum())
                    recall = nan_to_zero(tp / labels_np.sum())
                    f1 = nan_to_zero(2 * precision * recall / (precision + recall))
                ndcg = ndcg_score(labels_np.reshape(1, -1), preds_np.reshape(1, -1))
                all_scores.append((precision, recall, f1, ndcg))

                N = len(labels_np)
                K = int(labels_np.sum())

                sorted_indices = np.argsort(-preds_np)
                ranks = np.empty_like(sorted_indices)
                ranks[sorted_indices] = np.arange(1, N + 1)

                pos_mask = labels_np == 1
                S_obs = ranks[pos_mask].sum()

                S_best = K * (K + 1) / 2
                S_worst = K * (2 * N - K + 1) / 2

                pos_score = (S_worst - S_obs) / (S_worst - S_best)
                pos_score = float(np.clip(pos_score, 0.0, 1.0))

                position_scores.append(pos_score)

    print(f"Precision: {np.mean([x[0] for x in all_scores]):.3f}")
    print(f"Recall: {np.mean([x[1] for x in all_scores]):.3f}")
    f1 = np.mean([x[2] for x in all_scores])
    print(f"F1: {f1:.3f}")
    ndcg = np.mean([x[3] for x in all_scores])
    print(f"NDCG: {ndcg:.3f}")

    avg_position_avg = float(np.mean(position_scores))
    print(f"Average Position of Positive Sentences: {avg_position_avg:.3f}")

    if metric == "nDCG":
        return ndcg
    else:
        return avg_position_avg

def train_classifier(model_type, run_idx=None):
    model, tokenizer = load_model(model_type, "classification", num_labels=1)

    dataset = load_dataset(tokenizer)
    train_dataset = EvidenceSentenceDataset(dataset["train"], tokenizer)
    #train_loader = iter(DataLoader(train_dataset, batch_size=24, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)))

    weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = iter(DataLoader(
        train_dataset,
        batch_size=24,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    epochs_without_improvement = 0
    #criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion = torch.nn.BCEWithLogitsLoss()  # Remove the manual weighting since we're using sampling

    loss_avg = None

    best_score = evaluate_model(model, tokenizer, dataset, 'val', metric="position")
    print(f"Initial score: {best_score:.4f}")

    train_batch_generator = itertools.cycle(train_loader)

    for epoch in range(300):
        model.train()

        for _ in (bar := tqdm(range(200), desc=f"Epoch {epoch}", file=sys.stdout)):
            inputs, labels = next(train_batch_generator)
            outputs = model(**inputs)

            loss = criterion(outputs.logits.squeeze(-1), labels)

            #loss = criterion(outputs.logits.squeeze(-1), labels)
            #loss = (loss * labels + loss * (1 - labels) * 0.15).mean()

            if loss_avg is None:
                loss_avg = loss.item()
            else:
                loss_avg = 0.95 * loss_avg + 0.05 * loss.item()
            bar.desc = f"Epoch {epoch}, Loss: {loss_avg:.2f}"

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

            optimizer.step()
            optimizer.zero_grad()

        score = evaluate_model(model, tokenizer, dataset, 'val', metric="position")
        print(f"Validation score: {score:.4f}")

        if score > best_score:
            best_score = score
            save_model(model, f"{model_type}_EICAT_evidence{f'-{run_idx}' if run_idx is not None else ''}")
            epochs_without_improvement = 0
            print("New best! Model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 10:
            print("Early stopping triggered")
            break
        torch.cuda.empty_cache()

def test_model(model_type, run_index=None):
    model, tokenizer = load_model(model_type + f"_EICAT_evidence{f'-{run_index}' if run_index is not None else ''}", "classification", load_just_base_model=False, num_labels=1)
    dataset = load_dataset(tokenizer)
    return evaluate_model(model, tokenizer, dataset, "test")


def train_all_models(model_type):
    for run_idx in range(3):
        if not os.path.exists(data_dir(f"saved_models/model_{model_type}_EICAT_evidence-{run_idx}.pkl")):
            print(f"Training run {run_idx}")
            train_classifier(model_type, run_idx)

def test_all_models(model_type):
    scores = [[]]
    for run_idx in range(3):
        ndcg = test_model(model_type, run_idx)
        scores[0].append(ndcg)
    print(f"Average ndcg score: {float(np.mean(scores[0])):.3f}  (" + " ".join([f"{x:.3f}" for x in scores[0]]) + ")")
    return scores

if __name__ == "__main__":
    train_all_models("base")
    test_all_models("base")