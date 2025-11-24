import os.path
import random
import warnings
import sys
import numpy as np
import torch
from tqdm import tqdm
from evaluation.evaluate_predictions import evaluate_classification_predictions
from load_models_and_data.load_raw_biology_data import BioDataset
from load_models_and_data.load_models import load_model, save_model
from auxiliary.loss_fn import binary_cross_entropy_with_logits
from settings import device, data_dir

TRAIN_BATCH_SIZE = 8

def get_sample_idx(sampling_order, dataset):
    if len(sampling_order) == 0:
        sampling_order = [x for x in dataset.indices["train"]]
        random.shuffle(sampling_order)
    return sampling_order.pop()

def exp_avg(new_val, average, alpha=0.02):
    if average is None:
        average = new_val
    else:
        average = (1-alpha) * average + alpha * new_val
    return average

def evaluate_model(model, tokenizer, dataset, split, print_statistics=True):
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        for idx in tqdm(dataset.indices[split], desc="Evaluating...", position=0, leave=True, file=sys.stdout):
            sample = dataset.get_full_sample(idx)
            input = tokenizer(sample["prediction_text"], return_tensors='pt', truncation=True, max_length=512).to("cuda")

            model_output = model(**input)["logits"]
            prediction = (model_output > 0).float().reshape(1, -1)
            if torch.sum(prediction) == 0:
                prediction[0][torch.argmax(model_output)] = 1
            predictions.append(prediction.np())
            labels.append(sample["one_hot_label"])

    return evaluate_classification_predictions(np.concatenate(predictions, axis=0), np.stack(labels, axis=0), convert_predictions=False, print_statistics=print_statistics)

def train_classifier(model_type, run_idx=None):
    model, tokenizer = load_model(model_type, "classification")

    dataset = BioDataset(skip_unlabeled=True)

    label_sums = np.sum(np.stack([x["one_hot_label"] for x in dataset.samples.values() if x["index"] in dataset.indices["train"]], axis=0), axis=0)
    class_weights = 1 / torch.sqrt(torch.tensor(label_sums / len(dataset.indices["train"]), device=device))

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=4e-5, weight_decay=1e-4)
    loss_fn = binary_cross_entropy_with_logits

    print("Start training...")
    max_f1, _ = evaluate_model(model, tokenizer, dataset, "val")

    dataset.set_split("train")
    batches_per_epoch = max(int(len(dataset) / TRAIN_BATCH_SIZE), 50)
    sampling_order = []

    loss_avg = None
    epochs_without_improvement = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(300):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            bar = tqdm(desc="Loss: None", total=batches_per_epoch, position=0, leave=True, file=sys.stdout)

            for _ in range(batches_per_epoch):
                samples = [dataset.get_full_sample(get_sample_idx(sampling_order, dataset)) for _ in range(TRAIN_BATCH_SIZE)]
                input_texts = [x["prediction_text"] for x in samples]
                labels = torch.tensor([x["one_hot_label"] for x in samples], device=device)
                batch = tokenizer(input_texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

                model_output = model(**batch)["logits"]
                loss = loss_fn(model_output, labels, class_weights).mean()

                loss_avg = exp_avg(loss.item(), loss_avg)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                bar.update(1)
                bar.desc = f"Loss: {loss_avg:.3f}"

            bar.close()

            f1, _ = evaluate_model(model, tokenizer, dataset, "val")
            if f1 > max_f1 and f1 > 0.5:
                save_model(model, f"{model_type}_INAS_clf{f'-{run_idx}' if run_idx is not None else ''}")
                max_f1 = f1
                print("New best! Model saved.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement == 5:
                break
            torch.cuda.empty_cache()

    print(f"\nMax val f1 score: {max_f1}")
    if max_f1 < 0.5:
        return False
    return True

def test_model(model_type, run_index=None):
    model, tokenizer = load_model(model_type + f"_INAS_clf{f'-{run_index}' if run_index is not None else ''}", "classification", load_just_base_model=False)
    dataset = BioDataset(skip_unlabeled=True)
    return evaluate_model(model, tokenizer, dataset, "test")

def train_all_models(model_type):
    for run_idx in range(7):
        if not os.path.exists(data_dir(f"saved_models/model_{model_type}_INAS_clf-{run_idx}.pkl")):
            print(f"Training run {run_idx}")
            while not train_classifier(model_type, run_idx):
                pass

def test_all_models(model_type):
    macro_f1_scores, micro_f1_scores = [], []
    for run_idx in range(7):
        macro_f1, micro_f1 = test_model(model_type, run_idx)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
    print(f"Average macro score: {float(np.mean(macro_f1_scores)):.3f}  (" + " ".join([f"{x:.3f}" for x in macro_f1_scores]) + ")")
    print(f"Average micro score: {float(np.mean(micro_f1_scores)):.3f}  (" + " ".join([f"{x:.3f}" for x in micro_f1_scores]) + ")")
    return macro_f1_scores, micro_f1_scores