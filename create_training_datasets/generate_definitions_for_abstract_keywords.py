import os.path
from settings import data_dir
import json
from tqdm import tqdm
from load_models_and_data.load_raw_biology_data import BioDataset
import torch
from transformers import pipeline
from load_models_and_data.load_raw_biology_data import load_all_keywords
import re
from nltk.stem import PorterStemmer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ps = PorterStemmer()

os.makedirs(data_dir("processed_datasets/labeled_definitions"), exist_ok=True)
os.makedirs(data_dir("processed_datasets/unlabeled_definitions"), exist_ok=True)

def normalize_concept_name(concept):
    concept = concept.replace("-", " ")
    words = concept.split(" ")
    stemmed = [ps.stem(w) for w in words]
    return " ".join(stemmed)

def load_filenames(split):
    with open(data_dir("datasets/abstract_splits.json"), "r") as f:
        return json.load(f)[split]

def load_existing_definitions():
    filenames = load_filenames("5000")
    concepts = {}
    # Load unlabeled definitions
    for filename in filenames:
        if os.path.exists(data_dir(f"unlabeled_definitions/{int(filename)}.json")):
            with open(data_dir(f"unlabeled_definitions/{int(filename)}.json"), "r") as f:
                data = json.load(f)
                for key in data:
                    key_code = normalize_concept_name(key)
                    if key_code in concepts:
                        concepts[key_code]["definitions"].extend(data[key])
                        concepts[key_code]["concept_names"].append(key)
                    else:
                        concepts[key_code] = {"definitions": data[key],
                                              "concept_names": [key]}
    return {key: len(data["definitions"]) for key, data in concepts.items()}

def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype=torch.bfloat16)
    return pipe

def extract_generated_definition(output_string):
    pattern = r"Definition:\s*(.*?)(?=\s*END\.)"
    match = re.search(pattern, output_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def generate_continuations(text, keywords, pipe):
    all_definitions = {}

    with torch.no_grad():
        for keyword in keywords:
            prompt = f"""This is a scientific abstract:

                    {text}

                    Task: Generate a definition for "{keyword}" that contains its general meaning and is also slightly influenced by the content of the abstract.
                    Please do not use the word itself in its definition.

                    Format your response as:
                    Definition: [New Definition]
                    END.
                    """
            messages = [{"role": "user", "content": prompt}]

            response = pipe(messages, num_return_sequences=5, temperature=1.2, pad_token_id=pipe.tokenizer.eos_token_id)
            outputs_strings = [response[x]["generated_text"][1]["content"] for x in range(len(response))]
            definitions = [extract_generated_definition(x) for x in outputs_strings]
            all_definitions[keyword] = [x for x in definitions if x is not None]
    return all_definitions


def generate_definitions(start_idx, end_idx):
    dataset = BioDataset()
    keywords = load_all_keywords()
    pipe = load_model()

    existing_concept_definitions = load_existing_definitions()
    idx = 0
    bar = None
    for paper_idx in range(len(dataset.samples)):
        idx += 1
        if idx < start_idx or idx > end_idx:
            continue
        sample = dataset.samples[paper_idx]
        current_keywords = keywords[0][paper_idx]

        output_file = data_dir(f"processed_datasets/labeled_definitions/{sample['index']}.json")
        if os.path.exists(output_file):
            bar = None
            continue

        if bar is None:
            bar = tqdm(total=len(dataset.samples) + len(dataset.unlabeled_papers), initial=idx,
                       desc="Generating definitions...", smoothing=0.05)

        text = sample["prediction_text"]

        continuations = generate_continuations(text, current_keywords, pipe)
        if continuations is None:
            continue

        with open(output_file, "w") as f:
            json.dump(continuations, f)
        bar.update(1)

    for paper_idx in dataset.unlabeled_papers:
        idx += 1
        if idx < start_idx or idx > end_idx:
            continue

        if idx % 1000 == 0:
            existing_concept_definitions = load_existing_definitions()

        sample = dataset.unlabeled_papers[paper_idx]
        current_keywords = keywords[1][paper_idx]

        output_file = data_dir(f"unlabeled_definitions/{sample['index']}.json")
        if os.path.exists(output_file):
            with open(data_dir(f"unlabeled_definitions/{sample['index']}.json"), "r") as f:
                definitions = json.load(f)
        else:
            definitions = {}

        missing_definitions = [x for x in current_keywords if x not in definitions]
        missing_definitions = [x for x in missing_definitions if
                               normalize_concept_name(x) not in existing_concept_definitions or
                               existing_concept_definitions[normalize_concept_name(x)] < 20]

        if len(missing_definitions) == 0:
            continue

        if bar is None:
            bar = tqdm(total=len(dataset.samples) + len(dataset.unlabeled_papers), initial=idx,
                       desc="Generating definitions...", smoothing=0.05)

        text = sample["prediction_text"]

        continuations = generate_continuations(text, missing_definitions, pipe)
        if continuations is None:
            continue

        for concept in continuations:
            definitions[concept] = continuations[concept]

        with open(output_file, "w") as f:
            json.dump(definitions, f)
        bar.update(1)


if __name__ == '__main__':
    generate_definitions(0, 50000)