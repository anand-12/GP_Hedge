import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt.benchmarks import branin as _branin

from functools import partial
from skopt import gp_minimize, forest_minimize, dummy_minimize

def branin(x, noise_level=0.):
    return _branin(x) + noise_level * np.random.randn()

func = partial(branin, noise_level=2.0)
bounds = [(-5.0, 10.0), (0.0, 15.0)]
n_calls = 10

def run(minimizer, n_iter=5):
    return [minimizer(func, bounds, n_calls=n_calls, random_state=n)
            for n in range(n_iter)]

dummy_res = run(dummy_minimize)

gp_res = run(gp_minimize)



from skopt.plots import plot_convergence

plot = plot_convergence(("dummy_minimize", dummy_res),
                        ("gp_minimize", gp_res),
                        true_minimum=0.397887, yscale="log")

plot.legend(loc="best", prop={'size': 6}, numpoints=1)
plt.show()