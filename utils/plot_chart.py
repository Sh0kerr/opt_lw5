import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN = 0
MAX = 30
L = 1001


def plot_constraints(x1: np.ndarray, x2: np.ndarray) -> None:
    zhopa_1 = lambda x: x**2 + 2
    zhopa_2 = lambda x: -x + 6

    plt.plot(x1, zhopa_1(x1), color="red")
    plt.plot(x1, zhopa_2(x1), color="red")

    area = np.maximum(zhopa_1(x1), zhopa_2(x1))
    plt.fill_between(x1, zhopa_1(x1), area, color="pink", alpha=0.5)


def plot_contour(f: callable, x1: np.ndarray, x2: np.ndarray) -> None:
    x1, x2 = np.meshgrid(x1, x2)
    plt.contourf(x1, x2, f((x1, x2)), 1000, zorder=-1)


def plot_dots(dots: pd.Series) -> None:
    for i in range(len(dots)):
        plt.scatter(dots[i][0], dots[i][1], color="red")


def plot_dots_lines(dots: pd.Series) -> None:
    for i in range(len(dots) - 1):
        plt.plot([dots[i][0], dots[i + 1][0]], [dots[i][1], dots[i + 1][1]], color="blue")


def plot_chart(f: callable, dots: pd.Series) -> None:
    x1_arr = np.linspace(MIN, MAX, L)
    x2_arr = np.linspace(MIN, MAX, L)
    
    plot_constraints(x1_arr, x2_arr)
    plot_contour(f, x1_arr, x2_arr)
    plot_dots(dots)
    plot_dots_lines(dots)

    plt.title("Линии уровня функции F(x1, x2) и 'штрафная' область")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)

    plt.axhline(0, color="black")
    plt.axvline(0, color="black")

    plt.show()
