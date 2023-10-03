import numpy as np
import pandas as pd
import torch
import training
import validation
from custom_dataset import create_dataset
from init_wandb import Wandb_Init
from constants import Directories
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

"""
source:
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=dMZD0QCwnB6w
"""

device = "cuda" if cuda.is_available() else "cpu"


def read_training_dataset():
    df = pd.read_csv(
        Directories.TRAIN_SETS.joinpath("news_summary.csv"), encoding="latin-1"
    )
    df = df[["text", "ctext"]]
    df.ctext = "summarize: " + df.ctext
    print(df.head())
    return df


def save_model(model_output: str, model: tuple, save: bool = True) -> None:
    if save:
        model.save_pretrained(Directories.TRAINED_MODEL_DIR.joinpath(model_output))


def main(which_llm: str, model_output: str, train_epoch: int = 2):
    # start wandb
    wandb_init = Wandb_Init(model_output, train_epoch)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(wandb_init._config.SEED)
    np.random.seed(wandb_init._config.SEED)
    torch.backends.cudnn.deterministic = True

    # define tokenizer and model
    tokenizer: PreTrainedTokenizerBase = T5Tokenizer.from_pretrained(
        Directories.LLM_DIR.joinpath(which_llm)
    )
    model: PreTrainedModel = T5ForConditionalGeneration.from_pretrained(
        Directories.LLM_DIR.joinpath(which_llm)
    )
    model = model.to(device)  # send model to device

    # create a custom dataset
    # load them with DataLoader
    df = read_training_dataset()
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
        wandb_init, tokenizer, model, device, training_loader, optimizer
    )
    save_model(model_output, model, True)
    validation.start_validation(
        wandb_init, wandb_init._config.VAL_EPOCHS, tokenizer, model, device, val_loader
    )


if __name__ == "__main__":
    main("t5-small", train_epoch=1, model_output="trained-model-t5-small-test1")
