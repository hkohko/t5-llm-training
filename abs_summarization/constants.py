from dataclasses import dataclass
from pathlib import PurePath


@dataclass
class Directories:
    PROJ_DIR = PurePath(__file__).parents[0]
    TRAIN_SETS = PROJ_DIR.joinpath("training_set")
    LLM_DIR = PROJ_DIR.parents[2].joinpath("#LLM_MODELS")
    RESULT_DIR = PROJ_DIR.joinpath("model")
    TRAINED_MODEL_DIR = PROJ_DIR.joinpath("trained_model")
