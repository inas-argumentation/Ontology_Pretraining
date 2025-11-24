from auxiliary.visualize_text import visualize_word_importance
import editdistance
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def visualize_processed_annotation(annotation):
    visualize_word_importance(list([(1 if len(x["annotation"]) > 0 else 0, x["word"]) for x in annotation]))

label_matching = {
    "ER": 0,
    "BR": 1,
    "PH": 2,
    "DN": 3,
    "IS": 4,
    "LS": 5,
    "PP": 6,
    "DS": 7,
    "IM": 8,
    "TE": 9
}

def get_labels(annotations):
    labels = []
    for annotator in annotations:
        try:
            labels += [label_matching[x["type"][:2]] for x in annotations[annotator]["title annotations"].values()]
            labels += [label_matching[x["type"][:2]] for x in annotations[annotator]["abstract annotations"].values()]
        except:
            print("Label Error!")
    return list(set(labels))

def match_bert(annotation, text, label, offset=0):
    words = []
    max_idx = 0
    for a_idx, a in sorted([(x, y) for x, y in annotation.items() if label_matching[y["type"][:2]] == label], key=(lambda x: x[1]["char_span"][0])):
        if a["char_span"][0] < max_idx:
            if a["char_span"][1] <= max_idx:
                continue
            prev_a_idx = words[-1]["annotation"][0]
            for token in bert_tokenizer.tokenize(text[max_idx:a["char_span"][1]]):
                words.append({"word": token, "annotation": [prev_a_idx], "n_tokens": 1})
            max_idx = a["char_span"][1]
            continue
        for token in bert_tokenizer.tokenize(text[max_idx:a["char_span"][0]]):
            words.append({"word": token, "annotation": [], "n_tokens": 1})
        for token in bert_tokenizer.tokenize(text[a["char_span"][0]:a["char_span"][1]]):
            words.append({"word": token, "annotation": [int(a_idx)+offset], "n_tokens": 1})
        max_idx = a["char_span"][1]
        if len(words) > len(bert_tokenizer.tokenize(text[:max_idx])):
            print("Error!!!")
    for token in bert_tokenizer.tokenize(text[max_idx:]):
        words.append({"word": token, "annotation": [], "n_tokens": 1})
    w_idx = 0
    while w_idx < len(words):
        if words[w_idx]["word"][:2] == "##":
            words[w_idx-1]["word"] += words[w_idx]["word"][2:]
            words[w_idx-1]["n_tokens"] += 1
            del words[w_idx]
        else:
            w_idx += 1

    current_index = 0
    lower_text = text.lower()
    for idx in range(len(words)):
        found_idx = lower_text.find(words[idx]["word"], current_index)
        if found_idx == -1:
            raise Exception()
        words[idx]["word"] = text[found_idx:found_idx+len(words[idx]["word"])]
        current_index = found_idx + len(words[idx]["word"])
    return words

def match(annotation, tokenizer, text, label, offset=0, add_whitespace=False):
    bert_matching = match_bert(annotation, text, label, offset=offset)
    tokens = tokenizer.tokenize((" " if add_whitespace else "") + " ".join([x["word"] for x in bert_matching]))

    words = [{"word": t, "n_tokens": 1, "annotation": []} for t in tokens]
    words[0]["annotation"] = bert_matching[0]["annotation"]
    if words[0]["word"][0] == "Ġ":
        words[0]["word"] = words[0]["word"][1:]

    bert_matching_idx = 0
    idx = 1
    while idx < len(words):
        if words[idx]["word"][0] == "Ġ":
            bert_matching_idx += 1
            words[idx]["annotation"] = bert_matching[bert_matching_idx]["annotation"]
            words[idx]["word"] = words[idx]["word"][1:]
            idx += 1
        else:
            words[idx-1]["word"] += words[idx]["word"]
            words[idx-1]["n_tokens"] += 1
            del words[idx]

    annotation_indices = list(set([x for y in words for x in y["annotation"]]))
    annotated_spans = ["".join([item["word"] for item in words if int(idx) in item["annotation"]]) for idx in annotation_indices]
    for a_idx, a in sorted(
            [(x, y) for x, y in annotation.items() if label_matching[y["type"][:2]] == label],
            key=(lambda x: x[1]["char_span"][0])):
        start, end = a["char_span"]
        original_span = text[start:end]
        s = original_span.replace(" ", "")
        if not any([(s in x) for x  in annotated_spans]):
            print(original_span)


    return words


def match_test(annotation, tokenizer, text, label, offset=0):
    bert_matching = match_bert(annotation, text, label, offset=offset)
    tokens = tokenizer.tokenize(" ".join([x["word"] for x in bert_matching]))
    words = [{"word": t, "n_tokens": 1, "annotation": []} for t in tokens]

    idx = 1
    while idx < len(words):
        if words[idx]["word"][0] == "Ġ":
            words[idx]["word"] = words[idx]["word"][1:]
        else:
            if words[idx]["word"] not in ".!?,:;":
                words[idx-1]["word"] += words[idx]["word"]
                words[idx-1]["n_tokens"] += 1
                del words[idx]
                continue
        idx += 1
    if len(tokens) != sum([w["n_tokens"] for w in words]):
        raise Exception()

    annotation_char_spans = []
    for a_idx, a in sorted([(x, y) for x, y in annotation.items() if label_matching[y["type"][:2]] == label], key=(lambda x: x[1]["char_span"][0])):
        annotation_char_spans.append((a_idx, a["char_span"]))

    char_idx = 0
    word_idx = 0
    while char_idx < len(text):
        word = words[word_idx]["word"]
        add_idx = -1
        for idx in range(5):
            if text[char_idx+idx:char_idx+idx+len(word)] == word:
                add_idx = idx
                break
        if add_idx == -1:
            raise Exception()
        for ann_idx, ann_span in annotation_char_spans:
            if char_idx + add_idx in range(*ann_span) or (char_idx + add_idx + len(word) - 1 in range(*ann_span)):
                words[word_idx]["annotation"].append(ann_idx)

        char_idx += add_idx + len(word)
        word_idx += 1

    for a_idx, a in sorted(
            [(x, y) for x, y in annotation.items() if label_matching[y["type"][:2]] == label],
            key=(lambda x: x[1]["char_span"][0])
    ):
        annotated_words = [item["word"] for item in words if a_idx in item["annotation"]]
        reconstructed_text = " ".join(annotated_words)
        start, end = a["char_span"]
        original_span = text[start:end]

        def normalize_whitespace(s):
            s = ' '.join(s.split())
            for punct in ',.:;!?)]}':
                s = s.replace(f' {punct}', punct)
            for punct in '([{':
                s = s.replace(f'{punct} ', punct)
            return s

        normalized_reconstructed = normalize_whitespace(reconstructed_text)
        normalized_original = normalize_whitespace(original_span)
        if normalized_original != normalized_reconstructed and editdistance.eval(normalized_original, normalized_reconstructed) > 1:
            print(normalized_reconstructed)
            print(normalized_original)

    while len(overlap := [x["annotation"] for x in words if len(x["annotation"]) > 1]) > 0:
        idx_1 = overlap[0][0]
        idx_2 = overlap[0][1]
        for w_idx in range(len(words)):
            if idx_2 in words[w_idx]["annotation"]:
                words[w_idx]["annotation"] = list(set([x for x in words[w_idx]["annotation"] if x != idx_2] + [idx_1]))
    return words

def match_tokens_and_annotation(annotation, tokenizer, sample, label):
    title_annotations = match(annotation["title annotations"], tokenizer, sample["title"], label)
    abstract_annotations = match(annotation["abstract annotations"], tokenizer, sample["abstract"], label, len(annotation["title annotations"]), add_whitespace=True)
    return title_annotations + [{"word": ".", "annotation": [], "n_tokens": 1}] + abstract_annotations

def check_intersection(range1, range2):
    return not (range1[1] <= range2[0] or range2[1] <= range1[0])

def match_tokens_and_annotations_easy(annotation, tokenizer, sample, label):
    prediction_text = sample["prediction_text"]
    relevant_title_annotations = [a for a in annotation["title annotations"].values() if label_matching[a["type"][:2]] == label]
    relevant_abstract_annotations = [a for a in annotation["abstract annotations"].values() if label_matching[a["type"][:2]] == label]

    abstract_start_pos = prediction_text.find(sample["abstract"])
    if abstract_start_pos < 0:
        raise Exception()
    for a in relevant_abstract_annotations:
        a["char_span"] = [x + abstract_start_pos for x in a["char_span"]]
    relevant_annotations = relevant_title_annotations + relevant_abstract_annotations

    # Merge overlapping annotations
    merged = []
    if len(relevant_annotations) > 0:
        data = sorted(relevant_annotations, key=lambda x: x["char_span"][0])
        current = data[0]
        for item in data[1:]:
            current_span = current["char_span"]
            next_span = item["char_span"]

            if current_span[1] >= next_span[0]:
                current["char_span"] = (current_span[0], max(current_span[1], next_span[1]))
            else:
                merged.append(current)
                current = item
        merged.append(current)

    annotation_spans = {idx: a["char_span"] for idx, a in enumerate(merged)}

    text_tokenized = tokenizer(prediction_text, add_special_tokens=False)
    tokens = tokenizer.tokenize(prediction_text)
    words = text_tokenized.words()
    result = {i: {"word": "", "n_tokens": 0, "annotation": []} for i in range(max(words)+1)}

    for token_idx in range(len(tokens)):
        word = words[token_idx]
        result[word]["word"] += tokens[token_idx]
        result[word]["n_tokens"] += 1
        char_span = text_tokenized.token_to_chars(token_idx)
        for a_idx, annotation_span in annotation_spans.items():
            if check_intersection((char_span.start, char_span.end), annotation_span):
                result[word]["annotation"].append(a_idx)
                break

    for idx in result:
        result[idx]["annotation"] = list(set(result[idx]["annotation"]))
    return list(result.values())

def create_annotation_intersection(annotation_1, annotation_2):
    annotation = []
    idx_mapping = {}
    for w1, w2 in zip(annotation_1, annotation_2):
        if len(w1["annotation"]) > 0 and len(w2["annotation"]) > 0:
            s = str(w1["annotation"][0]) + str(w2["annotation"][0])
            if s not in idx_mapping:
                idx_mapping[s] = len(idx_mapping)
            a_idx = [idx_mapping[s]]
        else: a_idx = []
        annotation.append({"word": w1["word"], "annotation": a_idx, "n_tokens": w1["n_tokens"]})
    return annotation

def create_annotation_union(annotation_1, annotation_2):
    annotation = []
    prev_idx_1, prev_idx_2, current_a_idx = None, None, -1
    for w1, w2 in zip(annotation_1, annotation_2):
        idx_1 = None if len(w1["annotation"]) == 0 else w1["annotation"][0]
        idx_2 = None if len(w2["annotation"]) == 0 else w2["annotation"][0]
        if idx_1 is None and idx_2 is None:
            annotation.append({"word": w1["word"], "annotation": [], "n_tokens": w1["n_tokens"]})
        elif (None not in [prev_idx_1, idx_1] and prev_idx_1 == idx_1) or\
            (None not in [prev_idx_2, idx_2] and prev_idx_2 == idx_2):
            annotation.append({"word": w1["word"], "annotation": [current_a_idx], "n_tokens": w1["n_tokens"]})
        else:
            current_a_idx += 1
            annotation.append({"word": w1["word"], "annotation": [current_a_idx], "n_tokens": w1["n_tokens"]})
        prev_idx_1, prev_idx_2 = idx_1, idx_2
    return annotation

def create_gt_annotation(annotation, tokenizer, sample):
    annotations = {}
    labels = get_labels(annotation)
    for label in labels:
        label_annotations = []
        for annotator in annotation:
            label_annotations.append(match_tokens_and_annotations_easy(annotation[annotator], tokenizer, sample, label))
        num_annotators = len(annotation)
        if len(set([len(x) for x in label_annotations])) > 1:
            raise Exception("Tokenized text has different lengths for different annotators!")
        if num_annotators == 1:
            annotations[label] = label_annotations[0]
        elif num_annotators == 2: # Only for very rare cases, as usually only one or all three annotators annotated a given sample.
            annotations[label] = create_annotation_union(*label_annotations)
        elif num_annotators == 3:
            annotation_intersections = [create_annotation_intersection(label_annotations[x1], label_annotations[x2]) for x1, x2 in [[0, 1], [0, 2], [1, 2]]]
            annotations[label] = create_annotation_union(create_annotation_union(annotation_intersections[0], annotation_intersections[1]), annotation_intersections[2])
    for label in list(annotations.keys()):
        if len([x for x in annotations[label] if len(x["annotation"]) > 0]) == 0:
            del annotations[label]
    return annotations

def create_gt_annotations(annotations, tokenizer, dataset):
    new_annotations = {}
    from tqdm import tqdm
    for idx in tqdm(annotations):
        if not dataset.samples[int(idx)]["index"] == int(idx):
            raise Exception()
        new_annotations[int(idx)] = create_gt_annotation(annotations[idx], tokenizer, dataset.samples[int(idx)])
    return new_annotations