from sentence_transformers import SentenceTransformer
import numpy as np
from pso import PSO
from scipy.spatial.distance import pdist, squareform


def summarize(sentences, capacity=.5, n_iter=5000):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    similarity_matrix = 1 - squareform(pdist(sentence_embeddings, "cosine"))
    print(similarity_matrix)
    pso = PSO(n_particles=50, similarity_matrix=similarity_matrix, capacity=capacity)
    optimum = pso.optimize(n_iter)
    return '. '.join([sentences[i] for i in np.where(optimum==1)[0]])