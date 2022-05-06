from sentence_transformers import SentenceTransformer
import numpy as np
from pso import PSO
from scipy.spatial.distance import pdist, squareform


def summarize(sentences, n_iter=5000, length=4, capacity=.1):
    model = SentenceTransformer('all-MiniLM-L6-v2')
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