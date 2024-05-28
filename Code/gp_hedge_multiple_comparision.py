import numpy as np
import matplotlib.pyplot as plt
# from skopt.plots import plot_convergence
from skopt import gp_minimize
from skopt.benchmarks import hart6
from skopt.callbacks import VerboseCallback

noise_level = 1.0
bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)]
n_calls, n_random = 100, 5
initial_point_generator = "lhs"
# seeds = []
# for _ in range(10):
#     seeds.append(np.random.randint(1000))

seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

verbose_callback = VerboseCallback(n_total=n_calls)
callbacks = [verbose_callback]

results_pi = []
results_ei = []
results_lcb = []
results_gp_hedge = []

for seed in seeds:  
    results_pi.append(gp_minimize(hart6, bounds, base_estimator="GP", acq_func="PI",
                                  n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                                  initial_point_generator=initial_point_generator, random_state=seed,
                                  callback=callbacks))
    
    results_ei.append(gp_minimize(hart6, bounds, base_estimator="GP", acq_func="EI",
                                  n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                                  initial_point_generator=initial_point_generator, random_state=seed,
                                  callback=callbacks))
    
    results_lcb.append(gp_minimize(hart6, bounds, base_estimator="GP", acq_func="LCB",
                                   n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                                   initial_point_generator=initial_point_generator, random_state=seed,
                                   callback=callbacks))
    
    results_gp_hedge.append(gp_minimize(hart6, bounds, base_estimator="GP", acq_func="gp_hedge",
                                        n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                                        initial_point_generator=initial_point_generator, random_state=seed,
                                        callback=callbacks))

def plot_mean_convergence(results, label, ax, color):
    """Plot the mean convergence trace for a list of OptimizeResult."""
    n_calls = len(results[0].x_iters)
    iterations = range(1, n_calls + 1)
    mins = [[np.min(r.func_vals[:i]) for i in iterations] for r in results]
    mean_mins = np.mean(mins, axis=0)
    ax.plot(iterations, mean_mins, c=color, marker=".", markersize=12, lw=2, label=label)
    # for m in mins:
    #     ax.plot(iterations, m, c=color, alpha=0.2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Mean Convergence Plot")
ax.set_xlabel("Number of calls $n$")
ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
ax.grid()

colors = plt.cm.tab10(np.linspace(0.1, 1.0, 4))
plot_mean_convergence(results_pi, "pi_minimize", ax, colors[0])
plot_mean_convergence(results_ei, "ei_minimize", ax, colors[1])
plot_mean_convergence(results_lcb, "lcb_minimize", ax, colors[2])
plot_mean_convergence(results_gp_hedge, "gp_hedge_minimize", ax, colors[3])

ax.axhline(-3.32237, linestyle="--", color="r", lw=1, label="True minimum")
ax.legend(loc="best")

plt.show()
