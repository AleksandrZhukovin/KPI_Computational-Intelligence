import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from fitness_functions import *
from main import Optimizer


def plot_function(fitness_function, x1_domain, x2_domain, name):
    x1 = np.linspace(*x1_domain, 100)
    x2 = np.linspace(*x2_domain, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([fitness_function([x1, x2]) for x1, x2 in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title(name)
    plt.show()


def animate_search(optimizer, fitness_function, pop_size, x1_domain, x2_domain, name):
    pop = optimizer._initialize_population(pop_size)
    best_values, min_distances = [], []

    fig, ax = plt.subplots(figsize=(8, 6))
    x1 = np.linspace(*x1_domain, 100)
    x2 = np.linspace(*x2_domain, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([fitness_function([x1, x2]) for x1, x2 in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

    ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
    scatter = ax.scatter([], [], c='red')

    def update(frame):
        nonlocal pop
        ax.set_title(f'{name} {frame}')
        fitness = np.array([fitness_function(ind) for ind in pop])
        sorted_indices = np.argsort(fitness)
        best_values.append(fitness[sorted_indices[0]])

        selected = pop[sorted_indices[:pop_size // 2]]
        offspring = []

        while len(offspring) < pop_size:
            p1, p2 = selected[np.random.choice(len(selected), 2, replace=False)]
            crossover_point = np.random.randint(1, 2)
            child1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
            child2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))
            offspring.extend([child1, child2])

        pop = np.array(offspring[:pop_size])
        mutation = np.random.rand(pop_size, 2) < 0.01
        mutation_values = np.random.uniform(-0.5, 0.5, (pop_size, 2))
        pop += mutation * mutation_values
        pop[:, 0] = np.clip(pop[:, 0], *x1_domain)
        pop[:, 1] = np.clip(pop[:, 1], *x2_domain)

        scatter.set_offsets(pop)
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=optimizer._generations, interval=200, blit=False)
    filename = f'animations/{name}_pop{pop_size}.gif'
    os.makedirs("animations", exist_ok=True)
    ani.save(filename, writer='pillow', fps=10)
    plt.close(fig)


def plot_best_values(optimizer, fitness_function, x1_domain, x2_domain, name):
    pop_sizes = [25, 50, 100]
    plt.figure(figsize=(10, 5))

    for pop_size in pop_sizes:
        pop = optimizer._initialize_population(pop_size)
        best_values, _ = optimizer._genetic(pop, pop_size)
        best_values = np.array(best_values)
        plt.plot(range(len(best_values)), best_values.min(axis=1), label=f'Популяція {pop_size}')

    plt.xlabel('Покоління')
    plt.ylabel('Відстань')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()


functions = [
    ('Branin', branin, (-5, 10), (0, 15)),
    ('Easom', easom, (-100, 100), (-100, 100)),
    ('Goldstein-Price', goldstein_price, (-2, 2), (-2, 2)),
    ('Six-Hump Camel', six_hump_camel, (-3, 3), (-2, 2)),
]

for name, func, x1_domain, x2_domain in functions:
    optimizer = Optimizer(fitness_function=func, x1_domain=x1_domain, x2_domain=x2_domain)

    plot_function(func, x1_domain, x2_domain, name)

    # for pop_size in [25, 50, 100]:
    #     animate_search(optimizer, func, pop_size, x1_domain, x2_domain, name)

    plot_best_values(optimizer, func, x1_domain, x2_domain, name)
