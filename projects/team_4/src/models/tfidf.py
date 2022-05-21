
import re
import string
from spacy.lang.pl import Polish
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = Polish()

RE_EMOJI = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

class Tfidf:
    def __init__(self):
        self.vectorizer = None

    def preprocess_data(self, text):
        # remove whitespaces
        text = ' '.join(text.split())
        text = text.replace("\\\"", "")
        text = text.replace("\\n", "")
        # remove retweet tags
        if text.startswith("RT"):
            text = text[len("RT"):].lstrip()
        # remove response tags
        while text.startswith("@anonymized_account"):
            text = text[len("@anonymized_account"):].lstrip()
        while text.endswith("@anonymized_account"):
            text = text[:-len("@anonymized_account")].rstrip()
        # lower the text
        text = text.lower()
        # tokenize
        doc = [tok for tok in nlp(text)]
        # remove punctuation
        doc = [t for t in doc if t.text not in string.punctuation]
        # remove stop words
        doc = [tok for tok in doc if not tok.is_stop]
        return " ".join([x.text for x in doc])
    
    def fit_transform(self, x):
        self.vectorizer = TfidfVectorizer()
        return self.vectorizer.fit_transform(x).toarray()
    
    def preprocess_fit_transform(self, x):
        x_processed = [self.preprocess_data(xi) for xi in x]
        return self.fit_transform(x_processed)

    def transform(self, x):
        if self.vectorizer is None:
            raise NotFittedError("This TFIDF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.vectorizer.transform(x).toarray()
    
    def preprocess_transform(self, x):
        x_processed = [self.preprocess_data(xi) for xi in x]
        return self.transform(x_processed)
