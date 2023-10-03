import torch
from init_wandb import Wandb_Init
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.utils.data import DataLoader


def validate(
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
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


def start_validation(
    wandb_init: Wandb_Init,
    epoch: int,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: str,
    val_loader: DataLoader,
):
    print(
        "Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe"
    )
    for epoch in range(wandb_init._config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(
            "./models/predictions.csv"
        )  # Saving the dataframe as predictions.csv
        print("Output Files generated for review")
