import os.path
import sys
import time
import torch
from tqdm import tqdm
import random
import numpy as np
from load_models_and_data.load_EICAT_data import load_dataset
from auxiliary.loss_fn import categorical_cross_entropy_with_logits
from evaluation.evaluate_predictions import evaluate_classification_predictions
from blingfire import text_to_sentences
from collections import Counter
from settings import data_dir, device
from load_models_and_data.load_models import load_model, save_model

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def create_one_hot_label(labels, num_classes=6):
    one_hot = torch.zeros(num_classes)
    for label in labels:
        one_hot[label] = 1.0
    return one_hot

def create_sampled_text(text, tokenizer, max_length=500):
    sentences = list(enumerate(text_to_sentences(text).split("\n")))
    random.shuffle(sentences)
    n_tokens = 0
    sentence_selection = []
    while True:
        if len(sentences) > 0 and n_tokens + (new_tokens := len(tokenizer.tokenize(sentences[0][1]))) < max_length:
            sentence_selection.append(sentences.pop(0))
            n_tokens += new_tokens
        else:
            break
    return " ".join([x[1] for x in sorted(sentence_selection, key=lambda x: x[0])])

def pre_generate_compositions(dataset, split, tokenizer, n_compositions, seed=0, max_length=500):
    random.seed(seed)

    compositions = {}
    for idx, sample in enumerate(tqdm(dataset[split], desc=f"Pre-generating compositions for {split}", file=sys.stdout)):
        full_text = sample['text']
        sample_compositions = []

        for comp_idx in range(n_compositions):
            sampled_text = create_sampled_text(full_text, tokenizer, max_length)
            sample_compositions.append(sampled_text)

        compositions[idx] = sample_compositions

    return compositions

def evaluate_model(model, tokenizer, dataset, split, compositions):
    model.eval()
    all_predictions = []
    all_labels = []
    all_losses = []
    t = time.time()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset[split], desc=f"Evaluating {split}", file=sys.stdout)):
            sampled_predictions = []
            sampled_losses = []

            sample_compositions = compositions[idx]
            label = create_one_hot_label(sample['labels'])

            for sampled_text in sample_compositions:
                inputs = tokenizer(sampled_text, max_length=510, truncation=True, padding=True, return_tensors="pt").to(
                    "cuda")

                outputs = model(**inputs).logits[0]
                prediction = torch.argmax(outputs, dim=-1).item()
                sampled_predictions.append(prediction)
                sampled_losses.append(categorical_cross_entropy_with_logits(outputs, label.to("cuda")).item())

            # Add each prediction as a separate sample for F1 calculation
            for prediction in sampled_predictions:
                all_predictions.append(prediction)
                all_labels.append(label.numpy())

            # Average loss across all compositions for this sample
            all_losses.append(float(np.mean(sampled_losses)))

    macro_f1, micro_f1 = evaluate_classification_predictions(np.array(all_predictions), np.array(all_labels),
                                                             convert_predictions=True)
    avg_loss = np.mean(all_losses)
    print(f"{split} loss: {avg_loss:.3f}")
    print(time.time() - t)
    return avg_loss, macro_f1, micro_f1

def train_classifier(model_type, run_idx=None):
    model, tokenizer = load_model(model_type, "classification", num_labels=6)

    dataset = load_dataset(tokenizer)

    print("Pre-generating training compositions...")
    train_compositions = pre_generate_compositions(dataset, "train", tokenizer, n_compositions=10, seed=0)
    print("Pre-generating validation compositions...")
    val_compositions = pre_generate_compositions(dataset, "val", tokenizer, n_compositions=20, seed=1)

    label_counts = torch.stack([create_one_hot_label(x["labels"]) for x in dataset["train"]], dim=0).sum(0)
    class_weights = torch.tensor((1 / torch.sqrt(label_counts)) / (1 / torch.sqrt(label_counts)).mean(), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    epochs_without_improvement = 0
    batch_size = 32
    n_grad_acc_steps = 2
    current_accumulations = 0
    loss_avg = None

    score_func = lambda x: float(np.nan_to_num(x[1] + x[2] - x[0], nan=-np.inf))
    best_score = score_func(evaluate_model(model, tokenizer, dataset, 'val', val_compositions))
    print(f"Initial score: {best_score:.4f}")

    for epoch in range(300):
        model.train()
        total_loss = 0

        sample_indices = list(range(len(dataset['train'])))
        random.shuffle(sample_indices)

        batch_texts = []
        batch_labels = []
        for sample_idx in (bar := tqdm(sample_indices, desc=f"Epoch {epoch}", file=sys.stdout)):
            sample = dataset['train'][sample_idx]

            sampled_text = train_compositions[sample_idx][epoch % 10]
            batch_texts.append(sampled_text)
            label = create_one_hot_label(sample['labels']).to("cuda")
            batch_labels.append(label)

            if len(batch_texts) >= (batch_size // n_grad_acc_steps) or sample_idx == sample_indices[-1]:

                inputs = tokenizer(batch_texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
                outputs = model(**inputs).logits

                loss = (1 / n_grad_acc_steps) * categorical_cross_entropy_with_logits(outputs, torch.stack(batch_labels, dim=0), class_weights.unsqueeze(0)).mean()

                if loss_avg is None:
                    loss_avg = loss.item()
                else:
                    loss_avg = 0.95 * loss_avg + 0.05 * loss.item()
                bar.desc = f"Epoch {epoch}, Loss: {loss_avg:.2f}"

                loss.backward()
                current_accumulations += 1
                total_loss += loss.item()

                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

                if current_accumulations == n_grad_acc_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_accumulations = 0

                batch_texts = []
                batch_labels = []

        avg_loss = total_loss / len(dataset['train'])
        print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")

        epochs_without_improvement += 1
        score = score_func(evaluate_model(model, tokenizer, dataset, 'val', val_compositions))
        print(f"Validation score: {score:.4f}")

        if score > best_score:
            best_score = score
            save_model(model, f"{model_type}_EICAT_clf{f'-{run_idx}' if run_idx is not None else ''}")
            epochs_without_improvement = 0
            print("New best! Model saved.")

        if epochs_without_improvement >= 5:
            print("Early stopping triggered")
            break
        torch.cuda.empty_cache()
    if best_score > -1:
        return True
    else:
        return False

def test_model(model_type, run_index=None):
    model, tokenizer = load_model(model_type + f"_EICAT_clf{f'-{run_index}' if run_index is not None else ''}", "classification", load_just_base_model=False, num_labels=6)
    dataset = load_dataset(tokenizer)

    print("Pre-generating test compositions...")
    test_compositions = pre_generate_compositions(dataset, "test", tokenizer, n_compositions=20, seed=2)

    return evaluate_model(model, tokenizer, dataset, "test", test_compositions)


def train_all_models(model_type):
    for run_idx in range(7):
        if not os.path.exists(data_dir(f"saved_models/model_{model_type}_EICAT_clf-{run_idx}.pkl")):
            print(f"Training run {run_idx}")
            while not train_classifier(model_type, run_idx):
                pass

def test_all_models(model_type):
    macro_f1_scores, micro_f1_scores = [], []
    for run_idx in range(7):
        _, macro_f1, micro_f1 = test_model(model_type, run_idx)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
    print(f"Average macro score: {float(np.mean(macro_f1_scores)):.3f}  (" + " ".join([f"{x:.3f}" for x in macro_f1_scores]) + ")")
    print(f"Average micro score: {float(np.mean(micro_f1_scores)):.3f}  (" + " ".join([f"{x:.3f}" for x in micro_f1_scores]) + ")")
    return macro_f1_scores, micro_f1_scores