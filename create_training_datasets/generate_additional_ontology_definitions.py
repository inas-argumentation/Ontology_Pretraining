import re
from tqdm import tqdm
from settings import data_dir
import torch
from transformers import pipeline
import json
import os

GENERATED_DEFINITION_DATASET_PATH = data_dir("processed_datasets/ontology_definition_dataset.json")

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

def generate_definitions(sample, pipe, num_return_sequences=5):
    prompt = f"""Task: Create a single sentence that defines the concept listed below. You also receive an existing definition of the concept.
    If you feel like the definition does not contain enough information, please create a more extensive one. If you feel like all necessary information is
    already contained, you do not need to add additional information. Please do not simply repeat the definition given to you. Please do not use the term itself in the definition.

        Concept: {sample['concept']}
        Definition: {sample['definitions'][0] if len(sample['definitions']) > 0 else 'No definition available. Please define the concept on your own, and not that the concept shall be defined in the context of ecology.'}

        Format your response as:
        Definition: [New Definition]
        END.
        """
    messages = [{"role": "user", "content": prompt}]

    response = pipe(messages, num_return_sequences=num_return_sequences, temperature=1.2, pad_token_id=pipe.tokenizer.eos_token_id)
    outputs_strings = [response[x]["generated_text"][1]["content"] for x in range(len(response))]
    definitions = [extract_generated_definition(x) for x in outputs_strings]
    definitions = [x for x in definitions if x is not None]
    return definitions

def generate_definition_dataset():
    with open(GENERATED_DEFINITION_DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    number_of_definitions = 5
    pipe = load_model()

    for key in tqdm(dataset):
        sample = dataset[key]

        num_additional_definitions = number_of_definitions - len(sample["generated_definitions"])

        if num_additional_definitions == 0:
            continue

        definitions = generate_definitions(sample, pipe)
        dataset[key]["generated_definitions"] += definitions

        print(sample["concept"])
        print(definitions)

        with open(GENERATED_DEFINITION_DATASET_PATH, 'w') as f:
            json.dump(dataset, f, indent=2)

if __name__ == '__main__':
    generate_definition_dataset()