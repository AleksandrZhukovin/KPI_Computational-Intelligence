import numpy as np
import matplotlib

matplotlib.use('TkAgg')


class Optimizer:
    def __init__(self, *, fitness_function, bounds):
        self._fitness_function = fitness_function
        self._bounds = np.array(bounds)
        self._generations = 100
        self._dim = len(bounds)

    def _particle_swarm(self, pop_size):
        w = 0.5
        a1, a2 = 1.5, 1.5
        population = self._initialize_population(pop_size)
        velocity = np.random.uniform(-1, 1, (pop_size, self._dim))
        personal_best = population.copy()
        global_best = population[np.argmin([self._fitness_function(ind) for ind in population])]
        best_values = []

        for _ in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            best_values.append(population.copy())

            better_mask = fitness < np.array([self._fitness_function(ind) for ind in personal_best])
            personal_best[better_mask] = population[better_mask]

            if fitness.min() < self._fitness_function(global_best):
                global_best = population[np.argmin(fitness)]

            r1, r2 = np.random.rand(pop_size, self._dim), np.random.rand(pop_size, self._dim)
            velocity = w * velocity + a1 * r1 * (personal_best - population) + a2 * r2 * (global_best - population)
            population += velocity
            population = np.clip(population, self._bounds[:, 0], self._bounds[:, 1])

        return best_values

    def _initialize_population(self, pop_size):
        return np.random.uniform(self._bounds[:, 0], self._bounds[:, 1], (pop_size, self._dim))
