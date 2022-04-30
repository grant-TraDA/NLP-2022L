import numpy as np


class PSO:
    def __init__(self, n_particles, similarity_matrix) -> None:
        self.n_particles = n_particles
        self.similarity_matrix = similarity_matrix
        self.n_dim = self.similarity_matrix.shape[0]
        self.position = np.round(np.random.uniform(0, 1, (self.n_particles, self.n_dim)))
        self.velocity = np.random.uniform(0, 1, (self.n_particles, self.n_dim))
        self.pbest = self.position.copy()
        self.pbest_score = list(map(PSO.target_function, self.pbest))
        tmp_best = np.argmin(self.pbest_score)
        self.gbest = self.position[tmp_best,:].copy()
        self.gbest_score = self.pbest_score[tmp_best]

    @staticmethod
    def target_function(similarity_matrix, x):
        x = np.array(x).astype(bool)
        in_var = np.mean(similarity_matrix[x] ** 2)
        out_var = np.mean(similarity_matrix[x, x] ** 2)
        return out_var - in_var

    @staticmethod
    def constriction(phi):
        t1 = np.sqrt((phi ** 2) - (4 * phi))
        return 2 / np.abs(2 - phi - t1)

    def update_best(self):
        scores = list(map(PSO.target_function, self.position))
        for i, score in enumerate(scores):
            if score < self.pbest_score[i]:
                self.pbest_score[i] = score
                self.pbest[i] = self.position[i].copy()
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = self.position[i].copy()

    def update_velocity(self, constriction_coef, nostalgia_coef, social_coef):
        c1 = np.random.uniform(0, nostalgia_coef, self.position.shape)
        c2 = np.random.uniform(0, social_coef, self.position.shape)
        cognitive = (
            c1 * (self.pbest - self.position)
        )
        social = (
            c2 * (np.tile(self.gbest, (self.n_particles, 1)) - self.position)
        )
        self.velocity = constriction_coef * (self.velocity + cognitive + social)

    def update_position(self):
        sigmed = 1/(1 + np.exp(-self.velocity.copy()))
        rng = np.tile(np.random.uniform(0, 1, self.n_dim), (self.n_particles, 1))
        self.position = np.sign(sigmed-rng) / 2 + .5

    def optimize(self, n_iter) -> list[int]:
        for iter in range(n_iter):
            self.update_best()
            self.update_velocity(0.7298, 2.05, 2.05)
            self.update_position()
