import pandas as pd
import unicodedata
import re
from pathlib import Path
from typing import Union


def normalize_unicode(data: pd.DataFrame) -> pd.DataFrame:
    data["article"] = data["article"].apply(lambda x: unicodedata.normalize("NFKD", x))
    data["highlights"] = data["highlights"].apply(lambda x: unicodedata.normalize("NFKD", x))
    return data


def remove_noise(data: pd.DataFrame, col: str = "article") -> pd.DataFrame:
    data[col] = data[col].apply(lambda x: re.sub(r"(By \. [A-Za-z ]+ \. )", "", x))
    data[col] = data[col].apply(lambda x: re.sub(r"(PUBLISHED: \. \d{2}:\d{2} \w+, \d+ \w+ \d{4} \. )", "", x))
    data[col] = data[col].apply(lambda x: re.sub(r"(\| \. UPDATED: \. \d{2}:\d{2} \w+, \d+ \w+ \d{4} \. )", "", x))
    data[col] = data[col].apply(lambda x: re.sub(r"(\(CNN[A-Za-z ]*\)\s*-?-? ?)", "", x))
    return data


def drop_long_highlights(data: pd.DataFrame) -> pd.DataFrame:
    article_len = data["article"].apply(len)
    highlights_len = data["highlights"].apply(len)
    rows_to_drop = data[highlights_len > 0.5*article_len].index.values
    print(f"Droped {len(rows_to_drop)} rows with too long highlights")
    return data.drop(rows_to_drop, axis=0).reset_index(drop=True)


def preprocess(path: Union[str, Path]) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=[1, 2])
    data = data.drop_duplicates("highlights").reset_index(drop=True)
    data = data.pipe(normalize_unicode)\
               .pipe(remove_noise)\
               .pipe(drop_long_highlights)
    data["article"] = data["article"].apply(lambda x: re.split(r"[.?!;][ ']", x))
    return data


def save_data(data_dir: str) -> None:
    data_dir = Path(data_dir)
    for subset in ["train", "validation", "test"]:
        data = preprocess(data_dir / f"{subset}.csv")
        pkl_path = data_dir / f"{subset}.pkl.zip"
        data.to_pickle(pkl_path.as_posix())
        print(f"Saved file {pkl_path}")


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_pickle(data_dir / "train.pkl.zip")
    valid = pd.read_pickle(data_dir / "validation.pkl.zip")
    test = pd.read_pickle(data_dir / "test.pkl.zip")
    return train, valid, test

