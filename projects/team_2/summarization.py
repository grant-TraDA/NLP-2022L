import numpy as np
from p_tqdm import p_map
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer

from pso import PSO


class Summarizer(object):
    """Helper class allowing multiprocessing summarization runs.
    """
    def __init__(self, model: SentenceTransformer, n_iter: int=5000, length: int=4, capacity: float=.1) -> None:
        self.model: SentenceTransformer = model
        self.n_iter: int = n_iter
        self.length: int = length
        self.capacity: float = capacity

    def __call__(self, article: list[str]) -> str:
        return summarize(article, self.model, n_iter=self.n_iter, length=self.length, capacity=self.capacity)


def summarize_multiple(articles: list[list[str]], model: SentenceTransformer, n_iter: int=5000, length: int=4, capacity: float=.1) -> list[str]:
    return p_map(Summarizer(model, n_iter=n_iter, length=length, capacity=capacity), articles)


def summarize(sentences: list[str], model: SentenceTransformer, n_iter: int=5000, length: int=4, capacity: float=.1) -> str:
    sentence_embeddings: np.ndarray = model.encode(sentences)
    article_embedding: np.ndarray = model.encode('. '.join(sentences))
    sm: np.ndarray = 1 - squareform(
        pdist(np.concatenate((sentence_embeddings, article_embedding.reshape(1, -1)), axis=0), "cosine"))
    similarity_matrix: np.ndarray = sm[:-1, :-1]
    similarity__to_all: np.ndarray = sm[-1, :-1]
    pso: PSO = PSO(n_particles=50, similarities=(similarity_matrix, similarity__to_all), length=length, capacity=capacity)
    optimum: np.ndarray = pso.optimize(n_iter)
    return '. '.join([sentences[i] for i in np.where(optimum==1)[0]])
