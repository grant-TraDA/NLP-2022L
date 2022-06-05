
import sys
import os
sys.path.append(os.getcwd())

import pickle
import numpy as np

from models import roberta, tfidf
sys.modules['tfidf'] = tfidf
sys.modules['roberta'] = roberta


class PredictionService:

    class_dict = {
        0: "non-harmful",
        1: "cyberbullying",
        2: "hate-speech"
    }

    def __init__(
        self,
        clf_path="pretrained-models/combined-clf.pickle",
        tfidf_path="pretrained-models/tfidf.pickle",
        roberta_path="pretrained-models/roberta.pickle"
    ):
        with open(roberta_path, "rb") as handle:
            self.roberta = pickle.load(handle)
        with open(tfidf_path, "rb") as handle:
            self.tfidf = pickle.load(handle)
        with open(clf_path, "rb") as handle:
            self.clf = pickle.load(handle)

    def predict(self, text):
        bert_x = self.roberta.preprocess_extract_features([text])
        tfidf_x = self.tfidf.preprocess_transform([text])
        x = np.concatenate((bert_x,tfidf_x), axis=1)
        y = self.clf.predict(x)
        return self.class_dict[y[0]]
