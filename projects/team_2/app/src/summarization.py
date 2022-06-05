import numpy as np
from p_tqdm import p_map
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer

from src.pso import PSO


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
    """Summarizes many articles at once.

    Args:
        articles (list[list[str]]): list of articles in the form of list of sentences
        model (SentenceTransformer): embedding model from sentence_transformers package
        n_iter (int, optional): maximum number of iterations of the PSO algorithm. Defaults to 5000.
        length (int, optional): expected length of a summary. Defaults to 4.
        capacity (float, optional): parameter defining how much pso will use exceeding size penality, bigger capacity allows more exceeding summary length (0 - only penality, 1 - only similarity). Defaults to .1.

    Returns:
        list[str]: list of summaries
    """
    return p_map(Summarizer(model, n_iter=n_iter, length=length, capacity=capacity), articles)


def summarize(sentences: list[str], model: SentenceTransformer, n_iter: int=5000, length: int=4, capacity: float=.1) -> str:
    """Summarizes an article

    Args:
        sentences (list[str]): list of senteces
        model (SentenceTransformer): embedding model from sentence_transformers package
        n_iter (int, optional): maximum number of iterations of the PSO algorithm. Defaults to 5000.
        length (int, optional): expected length of a summary. Defaults to 4.
        capacity (float, optional): parameter defining how much pso will use exceeding size penality, bigger capacity allows more exceeding summary length (0 - only penality, 1 - only similarity). Defaults to .1.

    Returns:
        str: summary
    """
    # to calculate a similarity between sentences we use emeddings converting sentences to some vectors resembling sentences' contents
    sentence_embeddings: np.ndarray = model.encode(sentences)
    article_embedding: np.ndarray = model.encode('. '.join(sentences))
    # exact similarity is a cosine similarity between sentences' embeddings
    sm: np.ndarray = 1 - squareform(
        pdist(np.concatenate((sentence_embeddings, article_embedding.reshape(1, -1)), axis=0), "cosine"))
    similarity_matrix: np.ndarray = sm[:-1, :-1]
    similarity__to_all: np.ndarray = sm[-1, :-1]
    # here we run PSO on these embeddings
    pso: PSO = PSO(n_particles=50, similarities=(similarity_matrix, similarity__to_all), length=length, capacity=capacity)
    # in return we get 0-1 array indicating chosen sentences, which we then join to one string summarization
    optimum: np.ndarray = pso.optimize(n_iter)
    return '. '.join([sentences[i] for i in np.where(optimum==1)[0]])
