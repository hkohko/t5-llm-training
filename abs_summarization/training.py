from torch import long
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .init_wandb import Wandb_Init


def train(
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: str,
    loader: DataLoader,
    optimizer: Adam,
    wandb_init: Wandb_Init,
) -> None:
    model.train()

    for idx, data in enumerate(loader, 0):
        y = data.get("target_ids").to(device, dtype=long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data.get("source_ids").to(device, dtype=long)
        mask = data.get("source_mask").to(device, dtype=long)

        # https://github.com/priya-dwivedi/Deep-Learning/issues/137
        # lm_labels -> labels
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        loss = outputs[0]

        if idx % 10 == 0:
            try:
                wandb_init.wandb.log({"Training Loss": loss.item()})
            except RuntimeError:
                pass

        if idx % 500 == 0:
            try:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
            except RuntimeError:
                pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def start_training(
    wandb_init: Wandb_Init,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: str,
    training_loader: DataLoader,
    optimizer: Adam,
):
    print("Initiating Fine-Tuning for the model on our dataset")
    for epoch in range(wandb_init._config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer, wandb_init)
