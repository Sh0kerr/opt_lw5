import numpy as np


def numerical_gradient(func, mu, x, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pos = np.array(x, dtype=float)
        x_neg = np.array(x, dtype=float)
        x_pos[i] += epsilon
        x_neg[i] -= epsilon
        grad[i] = (func(x_pos, mu) - func(x_neg, mu)) / (2 * epsilon)
    return grad


def gradient_descent(func, mu, initial_point, learning_rate=0.01, tolerance=1e-6, max_iterations=1000, noisy=False):
    x = np.array(initial_point, dtype=float)
    for iteration in range(max_iterations):
        grad = numerical_gradient(func, mu, x)
        new_x = x - learning_rate * grad

        # Check for convergence
        if np.linalg.norm(new_x - x) < tolerance:
            print(f"Сошлось за {iteration + 1} итераций.") if noisy else None
            return new_x

        x = new_x

    print("Достигнуто максимальное количество итераций.") if noisy else None
    return x
