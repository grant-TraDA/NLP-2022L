import numpy as np


class PSO:
    def __init__(self, n_particles, similarities, length, capacity) -> None:
        self.n_particles = n_particles
        self.similarities = similarities
        self.length = length
        self.capacity = capacity
        self.n_dim = self.similarities[0].shape[0]
        self.position = np.round(np.random.uniform(0, 1, (self.n_particles, self.n_dim)))
        self.velocity = np.random.uniform(0, 1, (self.n_particles, self.n_dim))
        self.pbest = self.position.copy()
        self.pbest_score = [PSO._target_function(x, self.similarities, self.length, self.capacity) for x in self.pbest]
        tmp_best = np.argmin(self.pbest_score)
        self.gbest = self.position[tmp_best,:].copy()
        self.gbest_score = self.pbest_score[tmp_best]

    @staticmethod
    def _target_function(x, similarities, length, capacity):
        similarity_matrix = similarities[0]
        similarity_to_all = np.tile(similarities[1], (len(similarity_matrix), 1))
        sim_all = similarity_to_all + similarity_to_all.T - similarity_matrix
        x_triangle = (np.array(x).reshape(-1, 1) @ np.array(x).reshape(1, -1))
        x_triangle[np.tril_indices(len(x))] = 0
        return ((1 - capacity) * np.max([0, np.sum(x) - length])) - (capacity * np.sum(sim_all * x))

    @staticmethod
    def _constriction(phi):
        t1 = np.sqrt((phi ** 2) - (4 * phi))
        return 2 / np.abs(2 - phi - t1)

    def _update_best(self):
        scores = [PSO._target_function(x, self.similarities, self.length, self.capacity) for x in self.position]
        for i, score in enumerate(scores):
            if score < self.pbest_score[i]:
                self.pbest_score[i] = score
                self.pbest[i] = self.position[i].copy()
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = self.position[i].copy()

    def _update_velocity(self, constriction_coef, nostalgia_coef, social_coef):
        c1 = np.random.uniform(0, nostalgia_coef, self.position.shape)
        c2 = np.random.uniform(0, social_coef, self.position.shape)
        cognitive = (
            c1 * (self.pbest - self.position)
        )
        social = (
            c2 * (np.tile(self.gbest, (self.n_particles, 1)) - self.position)
        )
        self.velocity = constriction_coef * (self.velocity + cognitive + social)

    def _update_position(self):
        sigmed = 1/(1 + np.exp(-self.velocity.copy()))
        rng = np.tile(np.random.uniform(0, 1, self.n_dim), (self.n_particles, 1))
        self.position = np.sign(sigmed-rng) / 2 + .5

    def optimize(self, n_iter) -> list[int]:
        for iter in range(n_iter):
            self._update_best()
            self._update_velocity(0.7298, 2.05, 2.05)
            self._update_position()
        return self.gbest
