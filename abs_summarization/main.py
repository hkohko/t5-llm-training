import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration as ForConditionalGeneration
from transformers import MT5Tokenizer as Tokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizer

import abs_summarization.training as training
import abs_summarization.validation as validation
from .constants import *
from .custom_dataset import create_dataset
from .init_wandb import Wandb_Init

"""
source:
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=dMZD0QCwnB6w
"""


def read_training_dataset(file: str) -> pd.DataFrame:
    df = pd.read_csv(file, encoding=CSV_ENCODING.get(LANG))
    df = df[["text", "ctext"]]
    df.ctext = PREFIX_KEYS.get(LANG) + df.ctext
    return df


def save_model(model_output: str, model: PreTrainedModel) -> None:
    model.save_pretrained(Directories.TRAINED_MODEL_DIR.joinpath(model_output))


def main(which_llm: str, model_output: str, train_file: PurePath, fine_tuned_model: PurePath, train_epoch: int = 2):
    # start wandb
    wandb_init = Wandb_Init(model_output, train_epoch)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(wandb_init._config.SEED)
    np.random.seed(wandb_init._config.SEED)
    torch.backends.cudnn.deterministic = True

    # define tokenizer and model
    # https://huggingface.co/docs/transformers/model_doc/mt5#transformers.T5Tokenizer
    tokenizer: PreTrainedTokenizer = Tokenizer.from_pretrained(
        Directories.LLM_DIR.joinpath(which_llm)
    )
    model: PreTrainedModel = ForConditionalGeneration.from_pretrained(
        fine_tuned_model
    )
    model = model.to(DEVICE)  # send model to device

    # create a custom dataset
    # load them with DataLoader
    df = read_training_dataset(
        str(Directories.TRAIN_SETS_ID.joinpath(train_file))
    )
    training_set, val_set, train_params, val_params = create_dataset(
        wandb_init, df, tokenizer
    )
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer: Adam = torch.optim.Adam(
        params=model.parameters(), lr=wandb_init._config.LEARNING_RATE
    )

    # Log metrics with wandb
    wandb_init.wandb.watch(model, log="all")

    # train, save model, validate
    training.start_training(
        wandb_init, tokenizer, model, DEVICE, training_loader, optimizer
    )
    save_model(model_output, model)
    validation.start_validation(
        wandb_init, wandb_init._config.VAL_EPOCHS, tokenizer, model, DEVICE, val_loader
    )

def main_loop():
    model_name = ("finetuned-mt5-id")
    for idx, files in enumerate(Path(Directories.TRAIN_SETS_ID).iterdir()):
        finetuned_model = Directories.TRAINED_MODEL_DIR.joinpath(f"{model_name}-{idx - 1}")
        if idx == 0:
            finetuned_model = Directories.TRAINED_MODEL_DIR.joinpath("trained-mt5-ID-test")
        main("mt5-small", f"{model_name}-{idx}", files, finetuned_model, 3)
