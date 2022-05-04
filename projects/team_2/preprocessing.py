import pandas as pd
import unicodedata
import re
import string
from pathlib import Path
from typing import Union
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *


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


def remove_stopwords_and_stem(sentence: str, stemmer, stop_words, punctuation):
    sentence = sentence.translate(str.maketrans('', '', punctuation))
    sentence = sentence.lower()
    words = set(re.split(r"[ ']", sentence))
    new_words = [stemmer.stem(word) for word in words if word and word not in stop_words]
    return " ".join(new_words)


def preprocess(path: Union[str, Path], intense: bool = True) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=[1, 2])
    data = data.drop_duplicates("highlights").reset_index(drop=True)
    data = data.pipe(normalize_unicode)\
               .pipe(remove_noise)\
               .pipe(drop_long_highlights)
    data["article"] = data["article"].apply(lambda x: re.split(r"[.?!;][ ']", x))
    if intense:
        stopwords_english = set(stopwords.words('english'))
        punctuation = string.punctuation
        stemmer = PorterStemmer()
        data["article"] = data["article"].apply(
            lambda sentences: [remove_stopwords_and_stem(sentence, stemmer, stopwords_english, punctuation) for sentence in sentences])
    return data


def save_data(data_dir: str, subsets: list[str] = None, intense: bool = True) -> None:
    if subsets is None:
        subsets = ["train", "validation", "test"]
    data_dir = Path(data_dir)
    for subset in subsets:
        data = preprocess(data_dir / f"{subset}.csv", intense=intense)
        pkl_path = data_dir / f"{subset}.pkl.zip"
        data.to_pickle(pkl_path.as_posix())
        print(f"Saved file {pkl_path}")


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train, valid, test = None, None, None
    train_path = data_dir / "train.pkl.zip"
    valid_path = data_dir / "validation.pkl.zip"
    test_path = data_dir / "test.pkl.zip"
    if train_path.exists():
        train = pd.read_pickle(train_path)
    if valid_path.exists():
        valid = pd.read_pickle(valid_path)
    if test_path.exists():
        test = pd.read_pickle(test_path)
    return train, valid, test

