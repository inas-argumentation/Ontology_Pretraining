import torch
import os
from load_models_and_data.deberta_utils import DebertaForMaskedLM
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from settings import data_dir, device, Config

def to_np(self):
    return self.detach().cpu().numpy()
setattr(torch.Tensor, "np", to_np)

class SpanPredModel(torch.nn.Module):

    def __init__(self, model, n_labels=10):
        super(SpanPredModel, self).__init__()
        self.model = model
        self.n_labels = n_labels
        self.dense_1 = torch.nn.Linear(768, n_labels)
        self.dense_2 = torch.nn.Linear(768, n_labels)


    def forward(self, **kwargs):
        bert_out = self.model(**kwargs)["last_hidden_state"]

        text_clf_out = self.dense_1(bert_out[:,0])
        span_pred_out = self.dense_2(bert_out[:, 1:-1])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:2], self.n_labels)

        return text_clf_out, span_pred_out

def save_model(model, save_name):
    os.makedirs(data_dir("saved_models"), exist_ok=True)
    torch.save(model.state_dict(), data_dir(f"saved_models/model_{save_name}.pkl"))

def load_model(model_type, task_type, load_just_base_model=True, num_labels=10):
    if task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(Config.model_checkpoint, num_labels=num_labels)
    elif task_type == "span_prediction":
        model = AutoModel.from_pretrained(Config.model_checkpoint)
        model = SpanPredModel(model, num_labels)
    elif task_type == "MLM":
        if Config.model_checkpoint == "microsoft/deberta-base":
            model = DebertaForMaskedLM.from_pretrained(Config.model_checkpoint)
        else:
            model = AutoModelForMaskedLM.from_pretrained(Config.model_checkpoint)
    elif task_type == "SIM":
        model = AutoModel.from_pretrained(Config.model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    if model_type not in ["base", f"{Config.save_name}_base"]:
        if load_just_base_model:
            try:
                model.model.load_state_dict(torch.load(data_dir(f"saved_models/model_{model_type}.pkl")))
                print(f"Successfully loaded {data_dir(f"saved_models/model_{model_type}.pkl")}")
            except:
                model.deberta.load_state_dict(torch.load(data_dir(f"saved_models/model_{model_type}.pkl")))
                print(f"Successfully loaded {data_dir(f"saved_models/model_{model_type}.pkl")}")
        else:
            model.load_state_dict(torch.load(data_dir(f"saved_models/model_{model_type}.pkl")))
            print(f"Successfully loaded {data_dir(f"saved_models/model_{model_type}.pkl")}")
    model.to(device)
    return model, tokenizer

def check_model_checkpoint_available(model_type):
    return os.path.exists(data_dir(f"saved_models/model_{model_type}.pkl"))