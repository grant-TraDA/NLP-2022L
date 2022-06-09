from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


class HateSpeechNaiveBayes:
    def __init__(self, ngram_range=(1, 2), selector_percentile=5):
        self.ngram_range = ngram_range
        self.selector_percentile = selector_percentile
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.selector = SelectPercentile(f_classif, percentile=selector_percentile)
        self.model = MultinomialNB()

    def fit(self, data, labels):
        self.vectorizer.fit(data)
        embeddings = self.vectorizer.transform(data).toarray()
        self.selector.fit(embeddings, labels)
        selected_embeddings = self.selector.transform(embeddings)
        self.model.fit(selected_embeddings, labels)

    def predict(self, data):
        embeddings = self.vectorizer.transform(data).toarray()
        selected_embeddings = self.selector.transform(embeddings)
        results = self.model.predict(selected_embeddings)
        return results

    @property
    def label(self):
        return f'hate-speech-naive-bayes-ngram_range={self.ngram_range[1]}-selector_percentile={self.selector_percentile}'

