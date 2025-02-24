import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from fitness_functions import branin


matplotlib.use('TkAgg')


class Optimizer:
    def __init__(self, *, fitness_function, x1_domain, x2_domain):
        self._fitness_function = fitness_function
        self._x1_domain = x1_domain
        self._x2_domain = x2_domain
        self._generations = 100

    def _genetic(self, population, pop_size):
        best_values = []
        min_distances = []

        for gen in range(self._generations):
            fitness = np.array([self._fitness_function(ind) for ind in population])
            sorted_indices = np.argsort(fitness)
            best_values.append(fitness.copy())
            min_distances.append(abs(fitness[sorted_indices[0]]))

            selected = population[sorted_indices[:pop_size // 2]]
            offspring = []

            while len(offspring) < pop_size:
                p1, p2 = selected[np.random.choice(len(selected), 2, replace=False)]
                crossover_point = np.random.randint(1, 2)
                child1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
                child2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))
                offspring.extend([child1, child2])

            population = np.array(offspring[:pop_size])

            mutation = np.random.rand(pop_size, 2) < 0.01
            mutation_values = np.random.uniform(-0.5, 0.5, (pop_size, 2))
            population += mutation * mutation_values

            population[:, 0] = np.clip(population[:, 0], *self._x1_domain)
            population[:, 1] = np.clip(population[:, 1], *self._x2_domain)

        return best_values, min_distances

    def _initialize_population(self, pop_size):
        x1 = np.random.uniform(*self._x1_domain, pop_size)
        x2 = np.random.uniform(*self._x2_domain, pop_size)
        population = np.column_stack((x1, x2))
        return population

    def optimize(self, pop_size):
        pop = self._initialize_population(pop_size)
        res = self._genetic(pop, pop_size)
        return res
