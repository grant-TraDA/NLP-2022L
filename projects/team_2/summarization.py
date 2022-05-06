from sentence_transformers import SentenceTransformer
import numpy as np
from pso import PSO
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import multiprocessing


class Summarizer(object):
    def __init__(self, model, n_iter=5000, length=4, capacity=.1):
        self.model = model
        self.n_iter = n_iter
        self.length = length
        self.capacity = capacity

    def __call__(self, article):
        return summarize(article, self.model, n_iter=self.n_iter, length=self.length, capacity=self.capacity)


def summarize_multiple(articles, n_iter=5000, length=4, capacity=.1):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with multiprocessing.Pool() as pool:
        ret = pool.map(Summarizer(model, n_iter=n_iter, length=length, capacity=capacity), tqdm(articles))
    return ret


def summarize(sentences, model, n_iter=5000, length=4, capacity=.1):
    sentence_embeddings = model.encode(sentences)
    article_embedding = model.encode('. '.join(sentences))
    sm = 1 - squareform(
        pdist(np.concatenate((sentence_embeddings, article_embedding.reshape(1, -1)), axis=0), "cosine"))
    similarity_matrix = sm[:-1, :-1]
    similarity__to_all = sm[-1, :-1]
    pso = PSO(n_particles=50, similarities=(similarity_matrix, similarity__to_all), length=length, capacity=capacity)
    # print(pso.gbest_score)
    optimum = pso.optimize(n_iter)
    return '. '.join([sentences[i] for i in np.where(optimum==1)[0]])