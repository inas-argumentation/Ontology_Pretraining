from settings import data_dir
import json
import csv
from collections import defaultdict

ONTOLOGY_DATASET_PATH = data_dir("processed_datasets/ontology_definition_dataset.json")

def process_ontologies():
    try:
        with open(ONTOLOGY_DATASET_PATH, "r") as f:
            existing_definitions = json.load(f)
    except:
        existing_definitions = []

    label_to_definitions = defaultdict(list)
    parent_counts = defaultdict(int)
    parents = defaultdict(list)
    synonyms = defaultdict(list)
    label_to_synonyms = defaultdict(list)

    def remove_quotes(s):
        if len(s) == 0:
            return s
        if s[0] == "'" and s[-1] == "'":
            return s[1:-1]
        return s

    with open(data_dir("datasets/ontologies/INBIOV2.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        row_labels = next(reader)
        synonym_row = row_labels.index("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
        label_row = row_labels.index("http://www.w3.org/2000/01/rdf-schema#label")

        for row in reader:
            sample_labels = [x for x in [remove_quotes(x) for x in row[label_row].split("\t")] if len(x) > 2]
            sample_synonyms = [x for x in [remove_quotes(x) for x in row[synonym_row].split("\t")] if len(x) > 2]

            all_labels = list(set(sample_labels+sample_synonyms))
            for idx in range(len(all_labels)):
                synonyms[all_labels[idx].lower()] = all_labels[:idx] + all_labels[idx+1:]

    with open(data_dir("datasets/ontologies/INBIOV2 direct parent.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)

        for class_id, parent_id, class_label, parent_label, class_definition, parent_definition, class_comment, parent_comment in reader:
            if len(class_id) < 3 or len(class_label) < 3:
                continue

            if len(class_definition) > 10:
                label_to_definitions[class_label].append(class_definition)
            if len(label_to_definitions[class_label]) == 0 and len(class_comment) > 10:
                label_to_definitions[class_label].append(class_comment)

            if len(parent_label) > 2:
                parent_counts[parent_label] += 1

                parents[class_label].append(parent_label)

                if len(parent_definition) > 10:
                    label_to_definitions[parent_label].append(parent_definition)
                if len(label_to_definitions[parent_label]) == 0 and len(parent_comment) > 10:
                    label_to_definitions[parent_label].append(parent_comment)

    for x in label_to_definitions:
        label_to_definitions[x] = list(set(label_to_definitions[x]))
    for x in label_to_synonyms:
        label_to_synonyms[x] = list(set(label_to_synonyms[x]))
    for x in parents:
        parents[x] = list(set(parents[x]))

    final_inbio_dict = {}

    for concept in label_to_definitions:
        entry_dict = {"concept": concept,
                      "synonyms": synonyms[concept.lower()],
                      "definitions": label_to_definitions[concept],
                      "generated_definitions": [],
                      "parents": parents[concept],
                      "ontology": "INBIO"}
        generated_definitions = [x for x in existing_definitions if concept in x["concepts"]]
        generated_definitions = [x for x in generated_definitions if x["definition"] in label_to_definitions[concept]]
        if len(generated_definitions) > 0:
            entry_dict["generated_definitions"] = generated_definitions[0]["generated_definitions"]
        final_inbio_dict[concept] = entry_dict

    label_to_definitions = defaultdict(list)
    parent_counts = defaultdict(int)
    parents = defaultdict(list)
    synonyms = defaultdict(list)
    label_to_synonyms = defaultdict(list)

    with open(data_dir("datasets/ontologies/ENVO.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)

        for row in reader:
            id = row[0]
            sample_synonyms = row[2]
            if len(sample_synonyms) < 3:
                continue
            synonyms[id] += sample_synonyms.split("|")

    with open(data_dir("datasets/ontologies/ENVO direct parent.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)

        for class_id, parent_id, class_label, parent_label, class_definition, parent_definition, class_comment, parent_comment in reader:
            if len(class_id) < 3 or len(class_label) < 3:
                continue

            label_to_synonyms[class_label] += synonyms[class_id]
            label_to_synonyms[parent_label] += synonyms[parent_id]

            if len(class_definition) > 20:
                label_to_definitions[class_label].append(class_definition)
            if len(label_to_definitions[class_label]) == 0 and len(class_comment) > 20:
                label_to_definitions[class_label].append(class_comment)

            if len(parent_label) > 2:
                parent_counts[parent_label] += 1

                parents[class_label].append(parent_label)

                if len(parent_definition) > 20:
                    label_to_definitions[parent_label].append(parent_definition)
                if len(label_to_definitions[parent_label]) == 0 and len(parent_comment) > 20:
                    label_to_definitions[parent_label].append(parent_comment)

    for x in label_to_definitions:
        label_to_definitions[x] = list(set(label_to_definitions[x]))
    for x in label_to_synonyms:
        label_to_synonyms[x] = list(set(label_to_synonyms[x]))
    for x in parents:
        parents[x] = list(set(parents[x]))

    final_envo_dict = {}

    for concept in label_to_definitions:
        entry_dict = {"concept": concept,
                      "synonyms": label_to_synonyms[concept],
                      "definitions": label_to_definitions[concept],
                      "generated_definitions": [],
                      "parents": parents[concept],
                      "ontology": "ENVO"}
        generated_definitions = [x for x in existing_definitions if concept in x["concepts"]]
        generated_definitions = [x for x in generated_definitions if x["definition"] in label_to_definitions[concept]]
        if len(generated_definitions) > 0:
            entry_dict["generated_definitions"] = generated_definitions[0]["generated_definitions"]
        final_envo_dict[concept] = entry_dict

    complete_dict = final_envo_dict | final_inbio_dict

    with open(ONTOLOGY_DATASET_PATH, "w") as f:
        json.dump(complete_dict, f, indent=2)