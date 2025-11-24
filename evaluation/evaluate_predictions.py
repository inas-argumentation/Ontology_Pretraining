import numpy as np

from auxiliary.split_sample import split_sample_and_return_words

span_score_names = {0: "Average token auc",
                    1: "Average token F1",
                    2: "Discrete token F1",
                    3: "Average span IoU F1",
                    4: "Discrete span IoU F1"}

def evaluate_classification_predictions(y_pred, y_true, print_statistics=True, convert_predictions=True):
    num_classes = y_true.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        if convert_predictions:
            one_hot_pred = np.zeros((len(y_pred), num_classes))
            one_hot_pred[np.arange(len(y_pred)), y_pred] = 1

            one_hot_gt = y_true
        else:
            one_hot_pred = y_pred
            one_hot_gt = y_true

        gt_counts = np.sum(one_hot_gt, axis=0)
        predicted_counts = np.sum(one_hot_pred, axis=0)
        actually_there = (gt_counts > 0).astype("int32")

        tp = np.sum(one_hot_pred * one_hot_gt, axis=0)
        fp = np.sum(one_hot_pred * (1-one_hot_gt), axis=0)
        tn = np.sum((1-one_hot_pred) * (1-one_hot_gt), axis=0)
        fn = np.sum((1-one_hot_pred) * one_hot_gt, axis=0)

        micro_precision = tp.sum() / (tp.sum() + fp.sum())
        micro_recall = tp.sum() / (fn.sum() + tp.sum())
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        precision = tp / (tp + fp)
        recall = tp / (fn + tp)
        f1 = 2 * precision * recall / (precision + recall)
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0
        f1[np.isnan(f1)] = 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if print_statistics:
            print("\nPer class metrics:")
            print("  Class | Precision  |  Recall  |    F1   |  Accuracy  |  Count")
            for c in range(num_classes):
                if actually_there[c] > 0:
                    print(f"   {c:2d}   |   {precision[c]:.3f}    |  {recall[c]:.3f}   |  {f1[c]:.3f}  |   {accuracy[c]:.3f}    |  {gt_counts[c]}/{predicted_counts[c]}")
                else:
                    print(f"    -   |     -      |    -     |    -    |     -      |  {gt_counts[c]}/{predicted_counts[c]}")


        macro_f1 = np.sum(f1*actually_there) / np.sum(actually_there)
        if print_statistics:
            print(f"Macro F1: {macro_f1:.3f}  Micro F1: {micro_f1:.3f}")

    return macro_f1, micro_f1

from blingfire import text_to_sentences
from load_models_and_data.load_raw_biology_data import load_annotations_and_ground_truth
from sklearn.metrics import precision_recall_curve, auc
from auxiliary.visualize_text import visualize_word_importance

def calc_AUC_score(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    return score

def calc_F1(precision, recall):
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

def extract_spans(array, split_points=None):
    spans = []
    current_span = None
    for i in range(len(array)):
        if array[i] == 1:
            split = split_points is not None and i in split_points
            if current_span is None and not split:
                current_span = [i, i+1]
            elif not split:
                current_span[1] += 1
            elif current_span is not None:
                spans.append([current_span[0], current_span[1]+0])     # Omit punctuation? Otherwise +1
                current_span = None
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        spans.append(current_span)
    return spans

def calc_IoU_between_spans(spans_1, spans_2):
    IoU = np.zeros((len(spans_1), len(spans_2)), dtype="float32")
    for i in range(len(spans_1)):
        for j in range(len(spans_2)):
            max_min_val = max(spans_1[i][0], spans_2[j][0])
            min_max_val = min(spans_1[i][1], spans_2[j][1])
            n_overlap = float(max(0, min_max_val - max_min_val))
            n_union = float(len(set(list(range(spans_1[i][0], spans_1[i][1])) + list(range(spans_2[j][0], spans_2[j][1])))))
            IoU[i, j] = n_overlap / n_union
    return IoU

def calc_token_F1(pred, gt):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return calc_F1(precision, recall)

def calc_F1_and_span_F1_score(gt_spans, gt_array, pred, percentage, sentence_split_points, sorted_pred):
    threshold = sorted_pred[int((1-percentage)*pred.shape[0])]
    selection = (pred > threshold).astype('int')

    # Calculate continuous IoU F1 score
    pred_spans = extract_spans(selection, sentence_split_points)
    IoU = calc_IoU_between_spans(gt_spans, pred_spans)
    IoU_precision = np.mean(np.max(IoU, axis=0))
    IoU_recall = np.mean(np.max(IoU, axis=-1))
    IoU_F1 = calc_F1(IoU_precision, IoU_recall)

    # Calculate token F1 score
    token_F1 = calc_token_F1(selection, gt_array)

    return IoU_F1, token_F1

def calc_average_F1_and_span_F1_score(gt, pred, sample, tokenizer):
    pred = pred + np.linspace(0, 1, pred.shape[0]) * 1e-5

    # For Bio dataset, set last token of every sentence (punctuation) to zero, as annotated spans never cross sentence boundaries.
    sentences = text_to_sentences(sample["prediction_text"]).split("\n")

    sentence_spans = []
    current_index = 0
    for s in sentences:
        idx = sample["prediction_text"].find(s, current_index)
        if idx < 0:
            raise Exception()
        sentence_spans.append((idx, idx+len(s)))
        current_index = idx+len(s)

    sentence_end_indices = [x[1]-1 for x in sentence_spans]

    tokenizer_output = tokenizer(sample["prediction_text"], add_special_tokens=False)
    words = tokenizer_output.word_ids()
    for token_idx in range(len(words)):
        token_char_span = tokenizer_output.token_to_chars(token_idx)
        for end_idx in sentence_end_indices:
            if end_idx >= token_char_span.start and end_idx < token_char_span.end:
                pred[words[token_idx]] = 0

    if False:

        no_whitespace_sentences = [x.replace(" ", "") for x in sentences]

        words = split_sample_and_return_words(tokenizer, sample["prediction_text"])[0]
        if len(words) != len(gt):
            raise Exception()

        current_sentence_idx = 0
        current_char_idx = 0
        for idx, word in words.items():
            len_word = len(word["word"])
            if no_whitespace_sentences[current_sentence_idx][current_char_idx:current_char_idx+len_word] != word["word"]:
                print("Token not found!")
                raise Exception()
            current_char_idx += len_word
            if current_char_idx == len(no_whitespace_sentences[current_sentence_idx]):
                pred[idx] = 0
                current_sentence_idx += 1
                current_char_idx = 0

    #sentence_split_points = [num_words_per_sentence[0]-1]
    #for n in num_words_per_sentence[1:]:
    #    sentence_split_points.append(sentence_split_points[-1] + n)

    #for s in sentence_split_points:
    #    pred[s] = 0

    sentence_split_points = []
    gt_spans = extract_spans(gt)

    # Calculate discrete span and token F1 scores
    threshold = np.sort(pred)[-int(gt.sum())]
    #threshold = threshold - 0.2*np.sqrt(np.var(pred))
    selection = (pred >= threshold).astype("int")
    pred_spans = extract_spans(selection, sentence_split_points)
    if len(pred_spans) > 0:
        IoU = calc_IoU_between_spans(gt_spans, pred_spans)
        IoU = (IoU > 0.5).astype("int")
        tp = IoU.sum()
        precision = tp / len(pred_spans)
        recall = tp / len(gt_spans)
        discrete_IoU_F1 = calc_F1(precision, recall)
    else:
        discrete_IoU_F1 = 0
    discrete_token_F1 = calc_token_F1(selection, gt)

    # Calculate token F1 score and continuous span F1 score
    IoU_F1_scores = []
    token_F1_scores = []
    sorted_pred = np.sort(pred)
    for percentage in np.linspace(0.05, 0.95, 19):
        IoU_F1, token_F1 = calc_F1_and_span_F1_score(gt_spans, gt, pred, percentage, sentence_split_points, sorted_pred)
        IoU_F1_scores.append(IoU_F1)
        token_F1_scores.append(token_F1)
    return np.mean(token_F1_scores), discrete_token_F1, np.mean(IoU_F1_scores), discrete_IoU_F1

def evaluate_span_predictions(annotations, ground_truth_annotation_arrays, predictions, dataset, tokenizer, split="test"):
    #annotations, ground_truth_annotation_arrays = load_annotations_and_ground_truth(tokenizer, dataset)

    indices = dataset.indices[split]
    indices = [x for x in indices if x in annotations]

    scores = []
    for idx in indices:
        sample = dataset.get_full_sample(idx)
        gt = annotations[idx]

        current_sample_scores = []
        for label in gt:
            pred = predictions[idx][label]

            #pred = np.random.random(pred.shape)
            gt_array = np.array([1 if len(x["annotation"]) > 0 else 0 for x in gt[label]])
            current_sample_scores.append([calc_AUC_score(gt_array, pred), *calc_average_F1_and_span_F1_score(gt_array, pred, sample, tokenizer)])

        if False: # Uncomment to visualize predictions and annotations
            words = tokenizer.tokenize(sample["prediction_text"])
            i = 1
            while i < len(words):
                if words[i][:2] == "##":
                    words[i-1] += words[i][2:]
                    del words[i]
                else:
                    i += 1
            visualize_word_importance(list(zip(pred, gt_array, words)))
            print()
        if len(gt) > 0:
            scores.append(np.mean(np.array(current_sample_scores), axis=0))
            #print(sample["index"], scores[-1])

    mean_scores = np.mean(np.array(scores), axis=0)
    for idx in span_score_names:
        print(f"{span_score_names[idx]}:{' ' * (24 - len(span_score_names[idx]))} {mean_scores[idx]:.3f}")
    #print(f"Average auc score:                   {mean_scores[0]:.3f}")
    #print(f"Average span IoU F1 score:           {mean_scores[1]:.3f}")
    #print(f"Discrete span IoU F1 score:          {mean_scores[3]:.3f}")
    #print(f"Average token F1 score:              {mean_scores[2]:.3f}")
    #print(f"Discrete token F1 score:             {mean_scores[4]:.3f}")

    #print(f"{mean_scores[0]:.3f} & {mean_scores[2]:.3f} & {mean_scores[1]:.3f} & {mean_scores[4]:.3f} & {mean_scores[3]:.3f}")

    return mean_scores