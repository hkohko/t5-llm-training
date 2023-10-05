import json
from time import sleep
from pathlib import Path, PurePath

from transformers import MT5TokenizerFast

from abs_summarization.constants import Directories

TOKENIZER_FOLDER = PurePath(__file__).parents[0]
DATASET_FOLDER = TOKENIZER_FOLDER.joinpath("dataset")
SAVE_TOKENIZER = TOKENIZER_FOLDER.joinpath("trained_tokenizer")
mt5_tokenizer = MT5TokenizerFast.from_pretrained(
    Directories.LLM_DIR.joinpath("mt5-small")
)
mt5_updated_tokenizer = MT5TokenizerFast.from_pretrained(
    SAVE_TOKENIZER.joinpath("mt5-small-id-tokenizer")
)


def check():
    example2 = """
    Pada usia lima tahun, ia memulai debut aktingnya Robert Downey Sr. Filmnya Pound in 1970. pada tahun 1970. Dia kemudian bekerja dengan Brat Pack di film remaja Weird Science (1985) dan Less Than Zero (1987). Pada tahun 1992, Downey memerankan karakter judul dalam film biografi Chaplin, untuk itu dia dinominasikan untuk Aktor Terbaik dan memenangkan Penghargaan BAFTA. Mengikuti tugas di Corcoran Substance Abuse Treatment Facility atas tuduhan narkoba, dia bergabung dengan serial TV Ally McBeal, untuk itu dia memenangkan Penghargaan Golden Globe; namun setelah dua tuduhan narkoba, satu di akhir tahun 2000 dan satu di awal tahun 2001, dia dipecat dan karakternya dihentikan. Dia tetap dalam program perawatan narkoba yang diperintahkan pengadilan tidak lama setelah itu dan telah mempertahankan ketenangannya sejak 2003.
    """
    tokens = mt5_updated_tokenizer.tokenize(example2)
    print(tokens)


def give_dataset():
    for files in Path(DATASET_FOLDER).iterdir():
        with open(files) as file:
            for line in file:
                yield json.loads(line).get("text")
        sleep(10)


def train_tokenizer():
    new_tokenizer = mt5_tokenizer.train_new_from_iterator(give_dataset(), 520000)
    new_tokenizer.save_pretrained(SAVE_TOKENIZER.joinpath("mt5-small-id-tokenizer"))


if __name__ == "__main__":
    check()
