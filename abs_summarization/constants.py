from dataclasses import dataclass
from pathlib import PurePath

from torch import cuda


@dataclass
class Directories:
    PROJ_DIR = PurePath(__file__).parents[0]
    TRAIN_SETS = PROJ_DIR.joinpath("training_set")
    LLM_DIR = PROJ_DIR.parents[2].joinpath("#LLM_MODELS")
    RESULT_DIR = PROJ_DIR.joinpath("model")
    TRAINED_MODEL_DIR = PROJ_DIR.joinpath("trained_model")
    TRAIN_SETS_ID = TRAIN_SETS.joinpath("id_training_data")


DEVICE = "cuda" if cuda.is_available() else "cpu"
DEBUG = False

LANG = "ID"

PREFIX_KEYS = {"ID": "ringkaskan: ", "EN": "summarize: "}
TRAINING_SET_KEYS = {
    "ID": Directories.TRAIN_SETS_ID.joinpath("id_train.csv"),
    "EN": Directories.TRAIN_SETS.joinpath("news_summary.csv"),
}
CSV_ENCODING = {"ID": "utf-8", "EN": "latin-1"}
