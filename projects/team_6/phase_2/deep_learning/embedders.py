from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import spacy


class TfidfEmbedder:
    embedder = TfidfVectorizer()

    def fit(self, data, labels):
        self.embedder.fit(data)

    def transform(self, data):
        tokens = self.embedder.transform(data)
        return tokens

    @property
    def feature_count(self):
        return len(self.embedder.get_feature_names_out())

    @property
    def label(self):
        return "tfidf-vectorizer"

    @property
    def todense(self):
        return True


class CountEmbedder:
    def __init__(self, ngram_range=(1, 1), selector_percentile=3):
        self.ngram_range = ngram_range
        self.selector_percentile = selector_percentile
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.selector = SelectPercentile(f_classif, percentile=selector_percentile)

    def fit(self, data, labels):
        self.vectorizer.fit(data)
        embeddings = self.vectorizer.transform(data).toarray()
        self.selector.fit(embeddings, labels)

    def transform(self, data):
        embeddings = self.vectorizer.transform(data).toarray()
        selected_embeddings = self.selector.transform(embeddings)
        return selected_embeddings

    @property
    def feature_count(self):
        return len(self.selector.get_feature_names_out())

    @property
    def label(self):
        return f"count-vectorizer-ngram={self.ngram_range[1]}-selector={self.selector_percentile}"

    @property
    def todense(self):
        return False


class SpacyEmbedder:
    def __init__(self, corpus_type="pl_core_news_md"):
        self.corpus_type = corpus_type
        self.embedder = spacy.load(corpus_type)

    def fit(self, data, labels):
        return

    def transform(self, data):
        doc = self.embedder(data[0])  # done for custom dataset implementation
        vector = doc.vector
        return vector

    @property
    def feature_count(self):
        return 300

    @property
    def label(self):
        return f"spacy-vectorizer-{self.corpus_type}"

    @property
    def todense(self):
        return False