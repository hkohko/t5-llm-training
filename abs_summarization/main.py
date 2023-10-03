import numpy as np
import pandas as pd
import torch
from custom_dataset import create_dataset
from init_wandb import Wandb_Init
from constants import Directories
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

"""
source:
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=dMZD0QCwnB6w
"""

device = "cuda" if cuda.is_available() else "cpu"


def train(
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: tuple,
    device: str,
    loader: DataLoader,
    optimizer: Adam,
    wandb_init,
) -> None:
    model.train()
    for _, data in enumerate(loader, 0):
        y = data.get("target_ids").to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data.get("source_ids").to(device, dtype=torch.long)
        mask = data.get("source_mask").to(device, dtype=torch.long)

        # https://github.com/priya-dwivedi/Deep-Learning/issues/137
        # lm_labels -> labels
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        loss = outputs[0]

        if _ % 10 == 0:
            try:
                wandb_init.wandb.log({"Training Loss": loss.item()})
            except RuntimeError:
                pass

        if _ % 500 == 0:
            try:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
            except RuntimeError:
                pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: tuple,
    device: str,
    loader: DataLoader,
) -> tuple[list[str], list[str]]:
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]
            if _ % 100 == 0:
                print(f"Completed {_}")

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def read_training_dataset():
    df = pd.read_csv(
        Directories.TRAIN_SETS.joinpath("news_summary.csv"), encoding="latin-1"
    )
    df = df[["text", "ctext"]]
    df.ctext = "summarize: " + df.ctext
    print(df.head())
    return df


def start_validation(
    wandb_init,
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: tuple,
    device: str,
    val_loader: DataLoader,
):
    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print(
        "Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe"
    )
    for epoch in range(wandb_init._config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv("./models/predictions.csv")
        print("Output Files generated for review")


def start_training(
    wandb_init,
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: tuple,
    device: str,
    training_loader: DataLoader,
    optimizer: Adam,
    model_output: str,
):
    for epoch in range(wandb_init._config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer, wandb_init)


def save_model(save: bool = True) -> None:
    if save:
        model.save_pretrained(Directories.TRAINED_MODEL_DIR.joinpath(model_output))


def main(which_llm: str, model_output: str, train_epoch: int = 2):
    # WandB – Initialize a new run
    wandb_init = Wandb_Init(model_output, train_epoch)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(wandb_init._config.SEED)  # pytorch random seed
    np.random.seed(wandb_init._config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer: PreTrainedTokenizerBase = T5Tokenizer.from_pretrained(
        Directories.LLM_DIR.joinpath(which_llm)
    )
    df = read_training_dataset()

    training_set, val_set, train_params, val_params = create_dataset(
        wandb_init, df, tokenizer
    )

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model: tuple = T5ForConditionalGeneration.from_pretrained(
        Directories.LLM_DIR.joinpath(which_llm)
    )
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer: Adam = torch.optim.Adam(
        params=model.parameters(), lr=wandb_init._config.LEARNING_RATE
    )

    # Log metrics with wandb
    wandb_init.wandb.watch(model, log="all")
    # Training loop
    print("Initiating Fine-Tuning for the model on our dataset")
    start_training(
        wandb_init,
        train_epoch,
        tokenizer,
        model,
        device,
        training_loader,
        optimizer,
        model_output,
    )
    save_model(True)
    start_validation(wandb_init, epoch, tokenizer, model, device, val_loader)


if __name__ == "__main__":
    main("t5-small", train_epoch=2, model_output="trained-model-t5-small-test1")
