from torch.utils.data import Dataset
from torch import long


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        ctext = str(self.ctext[item])
        ctext = " ".join(ctext.split())

        text = str(self.text[item])
        text = " ".join(text.split())

        source = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=self.source_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.summ_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        source_ids = source.get("input_ids").squeeze()
        source_mask = source.get("attention_mask").squeeze()
        target_ids = target.get("input_ids").squeeze()
        target.get("attention_mask").squeeze()

        return {
            "source_ids": source_ids.to(dtype=long),
            "source_mask": source_mask.to(dtype=long),
            "target_ids": target_ids.to(dtype=long),
            "target_ids_y": target_ids.to(dtype=long),
        }


def create_dataset(wandb_init, df, tokenizer):
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation.
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=wandb_init._config.SEED)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(
        train_dataset,
        tokenizer,
        wandb_init._config.MAX_LEN,
        wandb_init._config.SUMMARY_LEN,
    )
    val_set = CustomDataset(
        val_dataset,
        tokenizer,
        wandb_init._config.MAX_LEN,
        wandb_init._config.SUMMARY_LEN,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": wandb_init._config.TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": wandb_init._config.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,
    }

    return training_set, val_set, train_params, val_params
