
import sys
import os
sys.path.append(os.getcwd())

import pickle
import numpy as np
from sklearn import svm

from roberta import Roberta
from tfidf import Tfidf

from helpers.stats import calculate_stats
from dataset.data_loader import DataLoader

if __name__ == "__main__":

    raw_train_x,train_y = DataLoader.load("training")
    raw_test_x,test_y = DataLoader.load("test")

    tfidf = Tfidf()
    train_tfidf_x = tfidf.preprocess_fit_transform(raw_train_x)
    test_tfidf_x = tfidf.preprocess_transform(raw_test_x)

    roberta = Roberta()
    try:
        train_bert_x = DataLoader.load_bert_features("train")
    except:
        print("Could not load train features from the hard drive. Calculating on the fly...")
        train_bert_x = roberta.preprocess_extract_features(raw_train_x)
    try:
        test_bert_x = DataLoader.load_bert_features("test")
    except:
        print("Could not load test features from the hard drive. Calculating on the fly...")
        test_bert_x = roberta.preprocess_extract_features(raw_test_x)

    train_x = np.concatenate((train_bert_x, train_tfidf_x), axis=1)
    test_x = np.concatenate((test_bert_x, test_tfidf_x), axis=1)

    clf = svm.LinearSVC(class_weight="balanced", max_iter=100_000, dual=True, verbose=1)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    stats = calculate_stats(test_y, predictions)
    for k,v in stats.items():
        print(f"{k}: {v}")

    if input("Type \"save\" to save the pretrained models (RoBERTa, TFIDF and Linear SVM) using pickle. It is required for further use of prediction service.\n") == "save":
        with open(f"pretrained-models/roberta.pickle", "wb") as handle:
            pickle.dump(roberta, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"pretrained-models/tfidf.pickle", "wb") as handle:
            pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"pretrained-models/combined-clf.pickle", "wb") as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
