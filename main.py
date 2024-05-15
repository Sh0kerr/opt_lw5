import pandas as pd

from funcs import f
from funcs.optimization import gradient_descent
from funcs.penalties.constraints import *
from funcs.penalties.penalty_function import penalty_function
from utils import plot_chart


def main() -> None:
    eps = 0.000001
    mu = 0.01
    x_0 = (0, 0)
    beta = 10
    p = 2
    # inequalities = None
    # inequalities = [inequality_3, inequality_4]
    # inequalities = [inequality_2, inequality_3, inequality_4]
    inequalities = [inequality_1, inequality_2, inequality_3, inequality_4]

    pen_func = lambda dot: penalty_function(dot, inequalities=inequalities, p=p)
    support_func = lambda dot, mu: f(dot) + mu * pen_func(dot)
    
    res = pd.DataFrame(columns=["k", "mu_k", "x_k+1", "f(x_k+1)", "alpha(x_mu_k)", "theta(mu_k)", "mu_k * alpha(x_mu_k)"])

    x_k = x_0
    for k in range(1, 1001):
        x_k = gradient_descent(support_func, mu, x_k, learning_rate=0.01, tolerance=1e-6, max_iterations=1000)

        res.loc[len(res.index)] = [k, mu, tuple(x_k), f(x_k), pen_func(x_k), support_func(x_k, mu), mu * pen_func(x_k)]

        if mu * pen_func(x_k) < eps:
            break
        
        mu *= beta

    res.index = res["k"]
    res.drop(columns=["k"], inplace=True)
    res.to_excel("data.xlsx")

    dots = pd.concat([pd.Series(data=[x_0], index=[0]), res["x_k+1"]])
    plot_chart(f, dots)


if __name__ == "__main__":
    main()
