import numpy as np
from typing import Tuple, List

class PSO:
    """Main class for performing PSO, which is part of the summarization process.
    """
    def __init__(self, n_particles: int, similarities: Tuple[np.ndarray, np.ndarray], length: int, capacity: float) -> None:
        """Initialization function saving model parameters and similarity matrices.

        Args:
            n_particles (int): number of paricles in PSO
            similarities (Tuple[np.ndarray, np.ndarray]): matrix of similarities between sentences and matrix of similarities of sentences to the whole article
            length (int): user defined length of summary in number of sentences
            capacity (float): parameter defining how much pso will use exceeding size penality, bigger capacity allows more exceeding summary length (0 - only penality, 1 - only similarity) 
            """        
        self.n_particles: int = n_particles
        self.similarities: Tuple[np.ndarray, np.ndarray] = similarities
        self.length: int = length
        self.capacity: float = capacity
        self.n_dim: int = self.similarities[0].shape[0]
        self.position: np.ndarray = np.round(np.random.uniform(0, 1, (self.n_particles, self.n_dim)))
        self.velocity: np.ndarray = np.random.uniform(0, 1, (self.n_particles, self.n_dim))
        # as we start we can assign particles current position as their historically best
        self.pbest: np.ndarray = self.position.copy()
        self.pbest_score: np.ndarray = [PSO._target_function(x, self.similarities, self.length, self.capacity) for x in self.pbest]
        # at this time we already seek globally best score among randomly chosen postions
        tmp_best: np.ndarray = np.argmin(self.pbest_score)
        self.gbest: np.ndarray = self.position[tmp_best,:].copy()
        self.gbest_score: np.ndarray = self.pbest_score[tmp_best]

    @staticmethod
    def _target_function(x: np.ndarray, similarities: Tuple[np.ndarray, np.ndarray], length: int, capacity: float) -> float:
        """Target function defining quality of solution found by the PSO.

        Args:
            x (np.ndarray): array of 0 and 1 indicating if a sentence should be included in the summary
            similarities (Tuple[np.ndarray, np.ndarray]): matrix of similarities between sentences and matrix of similarities of sentences to the whole article
            length (int): user defined length of summary in number of sentences
            capacity (float): parameter defining how much pso will use exceeding size penality, bigger capacity allows more exceeding summary length (0 - only penality, 1 - only similarity) 

        Returns:
            float: opposite of quality of solution x (the lower the better)
        """
        # similarity_matrix holds similarities of sentences between each other
        similarity_matrix: np.ndarray = similarities[0]
        # similarity_to_all holds similarities of sentences to the whole article
        similarity_to_all: np.ndarray = np.tile(similarities[1], (len(similarity_matrix), 1))
        sim_all: np.ndarray = similarity_to_all + similarity_to_all.T - similarity_matrix
        # depending on chosen subset of sentences x we create a mask to zero out similarities that we don't want to count
        x_triangle: np.ndarray = (np.array(x).reshape(-1, 1) @ np.array(x).reshape(1, -1))
        x_triangle[np.tril_indices(len(x))] = 0
        return ((1 - capacity) * np.max([0, np.sum(x_triangle) - length])) - (capacity * np.sum(sim_all * x_triangle))

    @staticmethod
    def _constriction(phi: float) -> float:
        """Calculating constriction coefficient based on Clerc's type 1 constriction (PSO specific).

        Args:
            phi (float): sum of nostalgia and social coefficient

        Returns:
            float: constriction coefficient
        """
        t1 = np.sqrt((phi ** 2) - (4 * phi))
        return 2 / np.abs(2 - phi - t1)

    def _update_best(self) -> None:
        """Calculates scores for new postions. If new positions are better than personal or global best updates them.
        """
        # we get scores for current positions of all particles
        scores: List[float] = [PSO._target_function(x, self.similarities, self.length, self.capacity) for x in self.position]
        for i, score in enumerate(scores):
            # updating personal best
            if score < self.pbest_score[i]:
                self.pbest_score[i] = score
                self.pbest[i] = self.position[i].copy()
            # updating global best
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = self.position[i].copy()

    def _update_velocity(self, nostalgia_coef: float, social_coef: float) -> None:
        """Calculates new velocities.

        Args:
            nostalgia_coef (float): how much particle wants to return to its personally best position
            social_coef (float): how much particle wants to go to the globally best position
        """
        # constriction coefficient depends on two other coefficients
        constriction_coef: float = PSO._constriction(nostalgia_coef + social_coef)
        # for every dimension in every particle's position we generate a random noise (craziness)
        c1: np.ndarray = np.random.uniform(0, nostalgia_coef, self.position.shape)
        c2: np.ndarray = np.random.uniform(0, social_coef, self.position.shape)
        # force bringing particle to its personal best
        cognitive: np.ndarray = (
            c1 * (self.pbest - self.position)
        )
        # force bringing particle to global best
        social: np.ndarray = (
            c2 * (np.tile(self.gbest, (self.n_particles, 1)) - self.position)
        )
        # actual velocity update
        self.velocity = constriction_coef * (self.velocity + cognitive + social)

    def _update_position(self) -> None:
        """Binarizes new velocity and updates positions.
        """
        # we convert velocity to the probabilty of position's element being 0  
        sigmed: np.ndarray = 1/(1 + np.exp(-self.velocity))
        # using aforementioned probabilites we sample for every dimension in every position
        rng: np.ndarray = np.tile(np.random.uniform(0, 1, self.n_dim), (self.n_particles, 1))
        # actual position update
        self.position = np.sign(sigmed-rng) / 2 + .5

    def optimize(self, n_iter: int, early_stopping: int = 20) -> np.ndarray:
        """Runs iterative PSO optimization algorithm.

        Args:
            n_iter (int): maximum number of iterations
            early_stopping (int): after how many iteration without improvement algorithm should stop. Use 0 to turn off early stopping.
        
        Returns:
            np.ndarray: optimal point represented as 0 and 1 array
        """
        # whole process is iterative
        previous_best: List[float] = []
        for _ in range(n_iter):
            # early stopping if PSO converged
            if early_stopping > 0 and len(previous_best) >= early_stopping and previous_best[-early_stopping] == previous_best[-1]:
                break

            self._update_best()
            # parameters below are chosen from broad literature and are commonly used in industry 
            self._update_velocity(2.05, 2.05)
            self._update_position()

            previous_best.append(self.gbest_score)
        return self.gbest
