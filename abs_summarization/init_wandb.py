import wandb


class Wandb_Init:
    def __init__(self, model_output: str, train_epoch: int):
        wandb.init(project=model_output)
        # WandB – Config is a variable that holds and saves hyperparameters and inputs
        # Defining some key variables that will be used later on in the training
        self._config = wandb.config  # Initialize config
        self._config.TRAIN_BATCH_SIZE = 2  # input batch size for training (default: 64)
        self._config.VALID_BATCH_SIZE = 2  # input batch size for testing (default: 1000)
        self._config.TRAIN_EPOCHS = train_epoch  # number of epochs to train (default: 10)
        self._config.VAL_EPOCHS = 1
        self._config.LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
        self._config.SEED = 42  # random seed (default: 42)
        self._config.MAX_LEN = 512
        self._config.SUMMARY_LEN = 150
