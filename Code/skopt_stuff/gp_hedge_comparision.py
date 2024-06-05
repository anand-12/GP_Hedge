import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from skopt import gp_minimize
from skopt.benchmarks import hart6
from skopt.callbacks import VerboseCallback


noise_level = 1.0
bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)]
n_calls, n_random, random_state = 10, 5, 31
initial_point_generator = "lhs"

verbose_callback = VerboseCallback(n_total=n_calls)
callbacks = [verbose_callback]


res_pi = gp_minimize(hart6, bounds, base_estimator="GP", acq_func="PI",
                     n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                     initial_point_generator=initial_point_generator, random_state=random_state,
                     callback=callbacks)

res_ei = gp_minimize(hart6, bounds, base_estimator="GP", acq_func="EI",
                     n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                     initial_point_generator=initial_point_generator, random_state=random_state,
                     callback=callbacks)

res_lcb = gp_minimize(hart6, bounds, base_estimator="GP", acq_func="LCB",
                      n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                      initial_point_generator=initial_point_generator, random_state=random_state,
                      callback=callbacks)

res_gp_hedge = gp_minimize(hart6, bounds, base_estimator="GP", acq_func="gp_hedge",
                           n_calls=n_calls, n_random_starts=n_random, noise=0.1**2,
                           initial_point_generator=initial_point_generator, random_state=random_state,
                           callback=callbacks)
print(res_pi.models)

fig, ax = plt.subplots(figsize=(10, 6))
ax, colors = plot_convergence(("pi_minimize", res_pi),
                 ("ei_minimize", res_ei),
                 ("lcb_minimize", res_lcb),
                 ("gp_hedge_minimize", res_gp_hedge),
                 true_minimum=-3.32237, ax=ax)


if hasattr(res_gp_hedge, 'index_for_gp_hedge'):
    acq_func_names = ['EI', 'LCB', 'PI']
    acq_func_colors = colors  
    acq_func_indices = res_gp_hedge.index_for_gp_hedge

    for i in range(n_random, n_calls): 
        start = i
        end = i + 1
        color = acq_func_colors[acq_func_indices[i - n_random]]
        ax.axvspan(start, end, facecolor=color, alpha=0.3)

handles = [plt.Line2D([0], [0], color=color, lw=4, label=label)
           for color, label in zip(acq_func_colors, acq_func_names)]

handles.append(plt.Line2D([0], [0], color='cyan', lw=4, label='gp_hedge'))

ax.legend(handles=handles, loc='best', prop={'size': 6})

plt.show()
