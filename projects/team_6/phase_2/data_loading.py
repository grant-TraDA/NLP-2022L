import os
import pandas as pd
from preprocessing import full_data_preprocessing, data_preprocessing

DATA_PATH = os.path.join(os.path.dirname(__file__), "../phase_2/data/")


def load_test():
    path = os.path.join(DATA_PATH, 'test_features.tsv')
    df = pd.read_csv(path, sep='\t')
    return df


def load_train():
    path = os.path.join(DATA_PATH, 'train.tsv')
    df = pd.read_csv(path, sep='\t')
    return df


def load_stopwords():
    path = os.path.join(DATA_PATH, 'stopwords.txt')
    df = pd.read_csv(path)
    return df


def load_full_data(stem=False):
    if stem:
        preprocessed_train_cache = os.path.join(DATA_PATH, "train_preprocessed_stem.tsv")
        preprocessed_test_cache = os.path.join(DATA_PATH, "test_preprocessed_stem.tsv")
    else:
        preprocessed_train_cache = os.path.join(DATA_PATH, "train_preprocessed.tsv")
        preprocessed_test_cache = os.path.join(DATA_PATH, "test_preprocessed.tsv")
    stopwords = load_stopwords()
    stopwords = data_preprocessing(stopwords)

    if not os.path.exists(preprocessed_train_cache):
        df_train = load_train()
        data_train = full_data_preprocessing(list(df_train['sentence']), stopwords, stem=stem)
        train_file_data = pd.DataFrame({"sentence": data_train, "target": list(df_train['target'])})
        train_file_data.to_csv(preprocessed_train_cache, sep="\t", index=False)
    else:
        df_train = pd.read_csv(preprocessed_train_cache, sep='\t')
        data_train = list(df_train['sentence'])

    if not os.path.exists(preprocessed_test_cache):
        df_test = load_test()
        data_test = full_data_preprocessing(list(df_test['sentence']), stopwords, stem=stem)
        train_test_data = pd.DataFrame({"sentence": data_test, "target": list(df_test['target'])})
        train_test_data.to_csv(preprocessed_test_cache, sep="\t", index=False)
    else:
        df_test = pd.read_csv(preprocessed_test_cache, sep='\t')
        data_test = list(df_test['sentence'])

    return {
        "train":
            {
                "data": data_train,
                "labels": list(df_train['target'])
            },
        "test":
            {
                "data": data_test,
                "labels": list(df_test['target'])
            }
        }
