import numpy as np
import itertools

def split_sample(tokenizer, text, max_number_of_tokens_per_split=510, array=None):
    text_tokenized = tokenizer.tokenize(text)
    tokenizer_out = tokenizer(text, add_special_tokens=False)

    n_overlap = 100

    n_splits = int(np.ceil(max(len(text_tokenized) - n_overlap, 1) / (max_number_of_tokens_per_split - n_overlap)))
    if n_splits == 1:
        splits = [list(range(len(text_tokenized)))]
        if array is not None:
            split_arrays = [array]
    else:
        tokens_per_split = int(np.ceil((len(text_tokenized) + (n_splits-1) * n_overlap) / n_splits))

        text_indices = list(range(len(text_tokenized)))

        splits = [text_indices[:tokens_per_split]]
        pos = tokens_per_split - n_overlap
        for i in range(1, n_splits):
            splits.append(text_indices[pos:pos + tokens_per_split])
            pos = pos + tokens_per_split - n_overlap

    words = create_word_list(tokenizer, text_tokenized, tokenizer_out, splits)
    if array is not None:
        split_arrays = [[] for _ in range(n_splits)]
        token_idx = 0
        for word in words.values():
            for split in word["splits"]:
                split_arrays[split].append(array[token_idx:token_idx+word["n_tokens"]])
            token_idx += word["n_tokens"]
        split_arrays = [np.concatenate(x) for x in split_arrays]

    split_texts = [tokenizer.convert_tokens_to_string(list(itertools.chain.from_iterable([x["tokens"] for x in words.values() if split in x["splits"]]))) for split in range(n_splits)]
    n_overlaps = [sum([x["n_tokens"] for x in words.values() if idx in x["splits"] and idx+1 in x["splits"]]) for idx in range(n_splits-1)]
    return split_texts, split_arrays if array is not None else [None], n_overlaps

def create_word_list(tokenizer, text_tokenized, tokenizer_out, splits):
    words = tokenizer_out.word_ids()
    try:
        whitespace_char = tokenizer.tokenize(" ")[0]
    except:
        whitespace_char = ""

    word_dict = {i: {"word": "", "n_tokens": 0, "tokens": [], "splits": set()} for i in range(max(words)+1)}

    for token_idx in range(len(text_tokenized)):
        word = words[token_idx]
        word_dict[word]["word"] += text_tokenized[token_idx].replace(whitespace_char, "")
        word_dict[word]["n_tokens"] += 1
        word_dict[word]["tokens"].append(text_tokenized[token_idx])
        token_splits = [x for x in range(len(splits)) if token_idx in splits[x]]
        if word_dict[word]["n_tokens"] > 1:
            word_dict[word]["splits"] = word_dict[word]["splits"].intersection(set(token_splits))
        else:
            word_dict[word]["splits"] = set(token_splits)

    return word_dict

def split_sample_and_return_words(tokenizer, text, max_number_of_tokens_per_split=510, n_overlap=100):
    text_tokenized = tokenizer.tokenize(text)
    tokenizer_out = tokenizer(text, add_special_tokens=False)

    n_splits = int(np.ceil(max(len(text_tokenized) - n_overlap, 1) / (max_number_of_tokens_per_split - n_overlap)))
    if n_splits == 1:
        splits = [list(range(len(text_tokenized)))]
    else:
        tokens_per_split = int(np.ceil((len(text_tokenized) + (n_splits-1) * n_overlap) / n_splits))

        text_indices = list(range(len(text_tokenized)))
        splits = [text_indices[:tokens_per_split]]
        pos = tokens_per_split - n_overlap
        for i in range(1, n_splits):
            splits.append(text_indices[pos:pos+tokens_per_split])
            pos = pos + tokens_per_split - n_overlap

    words = create_word_list(tokenizer, text_tokenized, tokenizer_out, splits)
    n_overlaps = [sum([x["n_tokens"] for x in words.values() if idx in x["splits"] and idx+1 in x["splits"]]) for idx in range(n_splits-1)]
    return words, len(splits), {i: sum([x["n_tokens"] for x in words.values() if i in x["splits"]]) for i in range(len(splits))}, n_overlaps