import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def animate_search(optimizer, pop_size, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(optimizer._bounds[0, 0], optimizer._bounds[0, 1], 100)
    y = np.linspace(optimizer._bounds[1, 0], optimizer._bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [[optimizer._fitness_function(np.array([x, y])) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    scatter = ax.scatter([], [], [], color='red', s=10)
    history = optimizer._particle_swarm(pop_size)

    def update(frame):
        points = history[frame]
        scatter._offsets3d = (points[:, 0], points[:, 1], [optimizer._fitness_function(p) for p in points])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=False)
    filename = f'animations/{name}_pop{pop_size}.gif'
    ani.save(filename, writer='pillow', fps=10)
    plt.close(fig)


def plot_best_values(optimizer, name):
    pop_sizes = [25, 50, 100]
    plt.figure(figsize=(10, 5))

    for pop_size in pop_sizes:
        best_values = optimizer._particle_swarm(pop_size)
        best_values = np.array(best_values)[:, :, 1]
        plt.plot(range(len(best_values)), best_values.min(axis=1), label=f'Популяція {pop_size}')

    plt.xlabel('Покоління')
    plt.ylabel('f(x)')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()


functions = [
    ('Ackley', ackley, [(-5, 5), (-5, 5)]),
    ('Rosenbrock', rosenbrock, [(-10, 10), (-10, 10)]),
    ('Cross-in-tray', cross_in_tray, [(-10, 10), (-10, 10)]),
    ('Hölder table', holder_table, [(-10, 10), (-10, 10)]),
    ('McCormick', mccormick, [(-1.5, 4), (-3, 4)]),
    ('Styblinski–Tang', styblinski_tang, [(-5, 5), (-5, 5)]),
]

for name, func, bounds in functions:
    optimizer = Optimizer(fitness_function=func, bounds=bounds)

    # plot_function(func, *bounds, name)

    # for pop_size in [25, 50, 100]:
    #     animate_search(optimizer, pop_size, name)

    plot_best_values(optimizer, name)
