import os.path
import json
import editdistance
from tqdm import tqdm
from load_models_and_data.load_raw_biology_data import BioDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from settings import data_dir

os.makedirs(data_dir("processed_datasets/labeled_keywords"), exist_ok=True)
os.makedirs(data_dir("processed_datasets/unlabeled_keywords"), exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"
prompts = ["Keywords:"]

def remove_substrings(strings):
    filtered_strings = []
    for i, s1 in enumerate(strings):
        is_substring = False
        for j, s2 in enumerate(strings):
            if i != j and s1 in s2:
                is_substring = True
                break
        if not is_substring:
            filtered_strings.append(s1)

    final_strings = []
    for s1 in filtered_strings:
        is_duplicate = False
        for s2 in final_strings:
            if editdistance.eval(s1, s2) < 4:
                is_duplicate = True
                break
        if not is_duplicate:
            final_strings.append(s1)

    return final_strings


def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", truncation_side="left")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.bfloat16)

    generation_config = GenerationConfig(
        max_new_tokens=30,
        num_beams=1,
        no_repeat_ngram_size=5,
        output_scores=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        top_k=3,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id)

    return model, tokenizer, generation_config

def generate_continuations(text, prompts, tokenizer, model, generation_config):
    all_keywords = set()
    with torch.no_grad():
        for prompt in prompts:
            input_batch = tokenizer([text + " " + prompt], return_tensors="pt", truncation=True, max_length=1300).to("cuda")
            #if input_batch["input_ids"].shape[1] > 1200:
            #    return None
            input_length = input_batch["input_ids"].shape[1]
            model_output = model.generate(**input_batch, generation_config=generation_config)

            idx = 0
            for output in model_output:
                idx += 1
                keyword_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                keyword_text = keyword_text.strip()
                if len(keyword_text) == 0:
                    continue
                if keyword_text[-1] == ".":
                    keyword_text = keyword_text[:-1]
                if "\n" in keyword_text:
                    keyword_text = keyword_text.split("\n")[0]
                keywords = [x for x in keyword_text.split(";")]
                keywords = [y for x in keywords for y in x.split(",")][:-1]
                keywords = [x.lower().strip() for x in keywords]
                all_keywords.update(keywords)

    all_keywords = remove_substrings(list(all_keywords))
    return all_keywords

def generate_keywords(start_idx, end_idx):
    dataset = BioDataset()
    model, tokenizer, generation_config = load_llama_model()

    idx = 0
    bar = None
    for paper_idx in range(len(dataset.samples)):
        idx += 1
        if idx < start_idx or idx > end_idx:
            continue
        sample = dataset.samples[paper_idx]

        output_file = data_dir(f"processed_datasets/labeled_keywords/{sample['index']}.json")
        if os.path.exists(output_file):
            bar = None
            continue

        if bar is None:
            bar = tqdm(total=len(dataset.samples)+len(dataset.unlabeled_papers), initial=idx, desc="Generating keywords...", smoothing=0.05)

        text = sample["prediction_text"]

        keywords = generate_continuations(text, prompts, tokenizer, model, generation_config)
        if keywords is None:
            continue

        with open(output_file, "w") as f:
            json.dump(keywords, f)
        bar.update(1)

    for paper_idx in dataset.unlabeled_papers:
        idx += 1
        if idx < start_idx or idx > end_idx:
            continue
        sample = dataset.unlabeled_papers[paper_idx]

        output_file = data_dir(f"processed_datasets/unlabeled_keywords/{sample['index']}.json")
        if os.path.exists(output_file):
            bar = None
            continue

        if bar is None:
            bar = tqdm(total=len(dataset.samples)+len(dataset.unlabeled_papers), initial=idx, desc="Generating keywords...", smoothing=0.05)

        text = sample["prediction_text"]

        keywords = generate_continuations(text, prompts, tokenizer, model, generation_config)
        if keywords is None:
            continue
        with open(output_file, "w") as f:
            json.dump(keywords, f)
        bar.update(1)

if __name__ == '__main__':
    generate_keywords(0, 80000)