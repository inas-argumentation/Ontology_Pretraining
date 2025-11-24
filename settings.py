import os

data_dir = lambda x: os.path.join(os.path.dirname(__file__), f"data/{x}")
device = "cuda"

class Config:
    save_name = None
    model_checkpoint = None

def set_save_name(save_name):
    Config.save_name = save_name

def set_model_checkpoint(checkpoint):
    Config.model_checkpoint = checkpoint

set_save_name("")
set_model_checkpoint("microsoft/deberta-base")