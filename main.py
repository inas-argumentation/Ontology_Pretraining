import os.path

from pretraining.model_pretraining import train_mixed
from settings import *
from evaluation import INAS_classification, INAS_span_prediction, EICAT_classification, EICAT_evidence_selection
from load_models_and_data.load_MLM_datasets import *
from load_models_and_data.load_SIM_datasets import *
import numpy as np

os.makedirs(data_dir("saved_models"), exist_ok=True)
os.makedirs(data_dir("training_data"), exist_ok=True)

task_dict = {"INAS_clf": INAS_classification,
             "INAS_span": INAS_span_prediction,
             "EICAT_clf": EICAT_classification,
             "EICAT_evidence": EICAT_evidence_selection}

score_names = {"INAS_clf": ["Macro F1", "Micro F1"],
               "INAS_span": ["AUC", "Token F1", "D Token F1", "IoU F1", "D IoU F1"],
               "EICAT_clf": ["Macro F1", "Micro F1"],
               "EICAT_evidence": ["NDCG"]}

def train_and_evaluate_pretraining_method(datasets, epochs=[200], tasks=["INAS_clf", "INAS_span", "EICAT_clf", "EICAT_evidence"], training_args=dict()):
    save_name = f"{Config.save_name}{'_MLM_' + datasets[0].dataset_name if datasets[0] is not None else ''}{'_SIM_' + datasets[1].dataset_name if datasets[1] is not None else ''}"

    if max(epochs) > 0 and not os.path.exists(data_dir(f"saved_models/model_{save_name}_epoch_{min([x for x in epochs if x > 0])}.pkl")):
        train_mixed(*datasets, save_epochs=epochs, **training_args)

    for task in tasks:
        for epoch in epochs:
            task_dict[task].train_all_models(f"{Config.save_name}_base" if epoch == 0 else f"{save_name}_epoch_{epoch}")

    results = {task: {} for task in tasks}
    for task in tasks:
        for epoch in epochs:
            model_type = f"{Config.save_name}_base" if epoch == 0 else f"{save_name}_epoch_{epoch}"
            results[task][epoch] =  task_dict[task].test_all_models(model_type)

    epoch_averages = {task: {i: [] for i in range(len(score_names[task]))} for task in tasks}
    for task in tasks:
        print(f"\nResults for task: {task}")
        for score_idx, score in enumerate(score_names[task]):
            print(f"Score: {score}")
            for epoch in epochs:
                avg_score = np.mean(results[task][epoch][score_idx])
                print(f"{epoch}: {avg_score:.3f}  ({' '.join([f'{x}:.3f' for x in results[task][epoch][score_idx]])})")
                epoch_averages[task][score_idx].append(avg_score)
            print(f"Overall average: {np.mean(epoch_averages[task][score_idx]):.3f}")

    print(Config.save_name)
    
def all_evaluations():
    set_model_checkpoint("microsoft/deberta-base")

    if False:
        # Evaluate DeBERTa without any further pretraining
        set_save_name("DeBERTa_test")
        train_and_evaluate_pretraining_method((None, None), epochs=[0])

    if False:
        # Just MLM on abstract sentences
        set_save_name("base_MLM_experiments")
        dataset_MLM = AbstractSentenceMLMDataset(split="5000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[200], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on ontology definitions (including LLM-generated ones)
        set_save_name("base_MLM_experiments")
        dataset_MLM = OntologyDefinitionMLMDataset(max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[40], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on abstract keyword definitions (LLM-generated ones)
        set_save_name("base_MLM_experiments")
        dataset_MLM = AbstractKeywordDefinitionMLMDataset(split="5000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[40], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on mix of ontology definitions (including LLM-generated ones) and abstract sentences
        set_save_name("base_MLM_experiments")
        dataset_MLM = MixedSentenceAndOntologyDefinitionMLMDataset(split="5000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[200], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on mix of abstract keyword definitions (LLM-generated ones) and abstract sentences
        set_save_name("base_MLM_experiments")
        dataset_MLM = MixedSentenceAndKeywordDefinitionMLMDataset(split="5000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[200], training_args={"weight_decay": 1e-2})

    if False:
        # Just SIM on ontology definitions
        set_save_name("base_SIM_experiments")
        dataset_SIM = OntologyDefinitionSIMDataset(use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((None, dataset_SIM), epochs=[40], training_args={"weight_decay": 0.0})

    if False:
        # Just SIM on abstract keyword definitions
        set_save_name("base_SIM_experiments")
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="5000", use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((None, dataset_SIM), epochs=[40], training_args={"weight_decay": 0.0})

    if False:
        set_save_name("base_MLM_plus_SIM_experiments")
        dataset_MLM = AbstractSentenceMLMDataset(split="5000", max_length=128)
        dataset_SIM = OntologyDefinitionSIMDataset(use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((dataset_MLM, dataset_SIM), epochs=[200], training_args={"weight_decay": 1e-2})

    if False:
        set_save_name("base_MLM_plus_SIM_experiments")
        dataset_MLM = AbstractSentenceMLMDataset(split="5000", max_length=128)
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="5000", use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((dataset_MLM, dataset_SIM), epochs=[200], training_args={"weight_decay": 1e-2})

    if False:
        set_save_name("PubMedBERT_test") # The model name was changed from PubMedBERT to BiomedBERT
        set_model_checkpoint("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        train_and_evaluate_pretraining_method((None, None), epochs=[0])

    if False:
        set_save_name("SciDeBERTa_test")
        set_model_checkpoint("KISTI-AI/Scideberta-full")
        train_and_evaluate_pretraining_method((None, None), epochs=[0])

    set_model_checkpoint("microsoft/deberta-base")

    if False:
        # Just MLM on larger dataset of abstract sentences (15000 abstracts)
        set_save_name("dataset_scaling_experiments_15000")
        dataset_MLM = AbstractSentenceMLMDataset(split="15000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[266], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on larger dataset of abstract sentences (25000 abstracts)
        set_save_name("dataset_scaling_experiments_25000")
        dataset_MLM = AbstractSentenceMLMDataset(split="25000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[333], training_args={"weight_decay": 1e-2})

    if False:
        # Just MLM on larger dataset of abstract sentences (35000 abstracts)
        set_save_name("dataset_scaling_experiments_35000")
        dataset_MLM = AbstractSentenceMLMDataset(split="35000", max_length=128)
        train_and_evaluate_pretraining_method((dataset_MLM, None), epochs=[400], training_args={"weight_decay": 1e-2})

    if False:
        # MLM on larger dataset of abstract sentences (15000 abstracts) combined with SIM loss on abstract keyword definitions
        set_save_name("dataset_scaling_experiments_15000")
        dataset_MLM = AbstractSentenceMLMDataset(split="15000", max_length=128)
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="15000", use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((dataset_MLM, dataset_SIM), epochs=[266], training_args={"weight_decay": 1e-2})

    if False:
        # MLM on larger dataset of abstract sentences (25000 abstracts) combined with SIM loss on abstract keyword definitions
        set_save_name("dataset_scaling_experiments_25000")
        dataset_MLM = AbstractSentenceMLMDataset(split="25000", max_length=128)
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="25000", use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((dataset_MLM, dataset_SIM), epochs=[333], training_args={"weight_decay": 1e-2})

    if False:
        # MLM on larger dataset of abstract sentences (35000 abstracts) combined with SIM loss on abstract keyword definitions
        set_save_name("dataset_scaling_experiments_35000")
        dataset_MLM = AbstractSentenceMLMDataset(split="35000", max_length=128)
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="35000", use_relatedness_loss=True)
        train_and_evaluate_pretraining_method((dataset_MLM, dataset_SIM), epochs=[400], training_args={"weight_decay": 1e-2})

    if False:
        # Ablation: SIM on abstract keyword definitions, but without relatedness between different concepts
        set_save_name("ablation_experiments")
        dataset_SIM = OntologyDefinitionSIMDataset(use_relatedness_loss=False)
        train_and_evaluate_pretraining_method((None, dataset_SIM), epochs=[40], training_args={"weight_decay": 0.0})

    if False:
        # Ablation: SIM on abstract keyword definitions, but without relatedness between different concepts
        set_save_name("ablation_experiments")
        dataset_SIM = AbstractKeywordDefinitionSIMDataset(split="5000", use_relatedness_loss=False)
        train_and_evaluate_pretraining_method((None, dataset_SIM), epochs=[40], training_args={"weight_decay": 0.0})

def generate_datasets():
    # The LLM-generated datasets already exist. If you want to regenerate them, delete the corresponding files in data/processed_datasets and run this method

    if False:
        # Convert raw ontology files into a json dataset file, which the LLM-generated definitions will be saved in
        from create_training_datasets import process_raw_ontology_files
        process_raw_ontology_files.process_ontologies()

    if False:
        # Generate additional definitions for all ontology concepts
        from create_training_datasets import generate_additional_ontology_definitions
        generate_additional_ontology_definitions.generate_definition_dataset()

    if False:
        # Extract keywords from scientific abstracts
        from create_training_datasets import generate_abstract_keywords
        generate_abstract_keywords.generate_keywords(0, 50_000)

    if False:
        # Generate definitions for extracted keywords
        from create_training_datasets import generate_definitions_for_abstract_keywords
        generate_definitions_for_abstract_keywords.generate_definitions(0, 50_000)

if __name__ == '__main__':
    # This function contains all calls necessary to regenerate the datasets used in our experiments.
    # Since they are already included, this step will not be necessary.
    generate_datasets()

    # This function contains all calls necessary to run the evaluation of baseline models and the pretraining experiments
    # with corresponding evaluations. Simply set the if check for the experiment that you want to run to True.
    all_evaluations()