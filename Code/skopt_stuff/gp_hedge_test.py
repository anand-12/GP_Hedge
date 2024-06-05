import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process, plot_gaussian_process_2D, plot_convergence
from skopt import gp_minimize
from test_functions import f, branin
from skopt.benchmarks import bench1
noise_level = 0.1
test_function = bench1



bounds = [(-100.0, 100.0)]
x = np.linspace(-100, 100, 400).reshape(-1, 1)
fx = [bench1([x_i]) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
        np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                        [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
        alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.show()


if test_function == f:
    bounds = [(0.0, 80.0)]
    x = np.linspace(0, 80, 200).reshape(-1, 1)
    fx = [f(x_i, noise_level=noise_level) for x_i in x]
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                            [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
            alpha=.2, fc="r", ec="None")
    plt.legend()
    plt.grid()
    plt.show()

elif test_function == branin:
    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    x1 = np.linspace(-5, 10, 400)
    x2 = np.linspace(0, 15, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([branin([x1, x2]) for x1, x2 in zip(np.ravel(X1), np.ravel(X2))]).reshape(X1.shape)

    # Plot the Branin function
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()



# Updated bounds to cover both dimensions of x
res = gp_minimize(bench1,                  # the function to minimize
                  [(-100., 100.)],  # the bounds on each dimension of x
                  acq_func="LCB",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  initial_point_generator="lhs", # random initialization
                  random_state=124   # the random seed
                )  


print(res.fun)
print(res.x)
plot_convergence(res)
plt.show()


# plt.rcParams["figure.figsize"] = (8, 14)


# def f_wo_noise(x):
#     return f(x, noise_level=noise_level)

# for n_iter in range(5):
#     # Plot true function.
#     plt.subplot(5, 2, 2*n_iter+1)

#     if n_iter == 0:
#         show_legend = True
#     else:
#         show_legend = False

#     ax = plot_gaussian_process(res, n_calls=n_iter,
#                                objective=f_wo_noise,
#                                noise_level=noise_level,
#                                show_legend=show_legend, show_title=False,
#                                show_next_point=False, show_acq_func=False)
#     ax.set_ylabel("")
#     ax.set_xlabel("")

#     plt.subplot(5, 2, 2*n_iter+2)
#     ax = plot_gaussian_process(res, n_calls=n_iter,
#                                show_legend=show_legend, show_title=False,
#                                show_mu=False, show_acq_func=True,
#                                show_observations=False,
#                                show_next_point=True)
#     ax.set_ylabel("")
#     ax.set_xlabel("")

# plt.show()

