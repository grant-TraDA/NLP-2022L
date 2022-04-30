from sentence_transformers import SentenceTransformer
import numpy as np
from pso import PSO

def _cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def summarize(sentences, capacity=.5, n_iter=5000):  
    n_sent = len(sentences)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    similarity_matrix = np.zeros((n_sent, n_sent))
    for i in range(n_sent):
        for j in range(i, n_sent):
            similarity = _cosine(sentence_embeddings[i], sentence_embeddings[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    print(similarity_matrix)
    pso = PSO(n_particles=50, similarity_matrix=similarity_matrix, capacity=capacity)
    optimum = pso.optimize(n_iter)
    return '. '.join([sentences[i] for i in np.where(optimum==1)[0]])