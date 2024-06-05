import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann, Ackley, Rosenbrock, Levy, Powell, Shekel, StyblinskiTang
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
import warnings
import pickle
import os
import random

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print(f"Using device: {device}")

# Define the true maxima and their locations for the test functions
true_maxima = {
    "Hartmann": 3.322368011415515,
    "Ackley": 0.0,
    "Rosenbrock": 0.0,
    "Levy": 0.0,
    "Powell": 0.0,
    "Shekel": 10.536409816115536,  # Corrected value from botorch documentation
    "StyblinskiTang": -39.16599 * 6  # For 6 dimensions
}

true_max_locations = {
    "Hartmann": torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
    "Ackley": torch.zeros(2),
    "Rosenbrock": torch.ones(4),
    "Levy": torch.ones(2),
    "Powell": torch.zeros(10),
    "Shekel": torch.tensor([4.0, 4.0, 4.0, 4.0]),  # Corrected value from botorch documentation
    "StyblinskiTang": torch.full((6,), -2.903534)  # For 6 dimensions
}


def target_function(test_function, individuals):
    result = []
    for x in individuals:
        result.append(-1.0 * test_function(x))
    return torch.tensor(result, dtype=dtype).to(device)

def generate_initial_data(test_function, n=10):
    print(f"Generating initial data with {n} points.")
    train_x = torch.rand(n, test_function.dim, dtype=dtype).to(device)
    exact_obj = target_function(test_function, train_x).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    print(f"Initial best observed value: {best_observed_value}")
    return train_x, exact_obj, best_observed_value

def fit_model(train_x, train_y, kernel_type):
    print(f"Fitting model with {kernel_type} kernel.")
    if kernel_type == 'RBF':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif kernel_type == 'Matern':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
    elif kernel_type == 'RQ':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    class CustomGP(SingleTaskGP):
        def __init__(self, train_x, train_y):
            super().__init__(train_x, train_y)
            self.covar_module = covar_module

    model = CustomGP(train_x, train_y).to(device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll

def calculate_weights(models, mlls, train_x, train_y):
    print("Calculating weights for models based on marginal log likelihood.")
    log_likelihoods = np.array([mll(models[i](train_x), train_y).sum().item() for i, mll in enumerate(mlls)])
    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    weights = np.exp(log_likelihoods) / np.sum(np.exp(log_likelihoods))
    print(f"Model weights: {weights}")
    return weights

def select_model(weights):
    print("Selecting model based on weights.")
    return np.random.choice(len(weights), p=weights)

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=['RBF', 'Matern', 'RQ'], acq_func_types=['EI', 'UCB', 'PI']):
    print("Getting next points.")
    models = []
    mlls = []
    for kernel in kernel_types:
        model, mll = fit_model(train_x, train_y, kernel)
        models.append(model)
        mlls.append(mll)

    weights = calculate_weights(models, mlls, train_x, train_y)
    selected_model_index = select_model(weights)
    selected_model = models[selected_model_index]
    print(f"Selected model index: {selected_model_index}")

    EI = ExpectedImprovement(model=selected_model, best_f=best_init_y)
    UCB = UpperConfidenceBound(model=selected_model, beta=0.1)
    PI = ProbabilityOfImprovement(model=selected_model, best_f=best_init_y)

    acq_funcs = {'EI': EI, 'UCB': UCB, 'PI': PI}
    acquisition_functions = [acq_funcs[acq] for acq in acq_func_types]

    candidates_list = []
    for acq_function in acquisition_functions:
        candidates, _ = optimize_acqf(acq_function=acq_function, bounds=bounds, q=n_points, num_restarts=20, raw_samples=512, options={"batch_limit": 5, "maxiter": 200})
        candidates_list.append(candidates)

    logits = np.array(gains)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=probs)
    print(f"Chosen acquisition function index: {chosen_acq_index}")

    return candidates_list[chosen_acq_index], chosen_acq_index, selected_model_index, selected_model

def update_data(train_x, train_y, new_x, new_y):
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    return train_x, train_y

def run_experiment(test_function, n_iterations, kernel_types, acq_func_types, n_runs, seeds, bounds):
    all_best_observed_values = []

    for run, seed in enumerate(seeds):
        print(f"Run number {run + 1}/{n_runs} with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        init_x, init_y, best_init_y = generate_initial_data(test_function, 1)
        gains = np.zeros(len(acq_func_types))
        eta = 0.1  

        best_observed_values = []

        train_x, train_y = init_x, init_y
        best_init_y = train_y.max().item()

        for t in range(n_iterations):
            print(f"Iteration number {t + 1}/{n_iterations}")
            print(f"Best function value yet: {best_init_y}")
            new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(train_x, train_y, best_init_y, bounds, eta, 1, gains, kernel_types, acq_func_types)
            new_results = target_function(test_function, new_candidates).unsqueeze(-1)

            train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

            best_init_y = train_y.max().item()
            best_observed_values.append(best_init_y)

            posterior_mean = selected_model.posterior(new_candidates).mean
            reward = posterior_mean.mean().item()
            gains[chosen_acq_index] += reward

        all_best_observed_values.append(best_observed_values)

        # Save each run's results
        result_file = f"{test_function.__class__.__name__}_run_{run + 1}_seed_{seed}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(best_observed_values, f)
        print(f"Saved results to {result_file}")

    mean_best_observed_values = np.mean(all_best_observed_values, axis=0)
    return mean_best_observed_values, all_best_observed_values

def plot_results(mean_results, titles, test_function_name, save_path, true_maximum, true_max_location):
    plt.figure(figsize=(12, 8))

    for mean_best_observed_values, title in zip(mean_results, titles):
        plt.plot(mean_best_observed_values, marker='.', linestyle='-', label=title)
    
    plt.axhline(y=true_maximum, color='k', linestyle='--', label='True Maxima')
    plt.title(f"Mean Performance Comparison for {test_function_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

    if true_max_location is not None:
        print(f"Global maximum location for {test_function_name}: {true_max_location.cpu().numpy()}")

n_iterations = 3
n_runs = 2
seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

# Define test functions, bounds and true maxima
test_functions = [
    (Hartmann(dim=6).to(device, dtype=dtype), torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]], dtype=dtype).to(device), true_maxima["Hartmann"], true_max_locations["Hartmann"]),
    (Ackley(dim=2).to(device, dtype=dtype), torch.tensor([[-32.768, -32.768], [32.768, 32.768]], dtype=dtype).to(device), true_maxima["Ackley"], true_max_locations["Ackley"]),
    # (Rosenbrock(dim=4).to(device, dtype=dtype), torch.tensor([[-5., -5., -5., -5.], [10., 10., 10., 10.]], dtype=dtype).to(device), true_maxima["Rosenbrock"], true_max_locations["Rosenbrock"]),
    (Levy(dim=2).to(device, dtype=dtype), torch.tensor([[-10., -10.], [10., 10.]], dtype=dtype).to(device), true_maxima["Levy"], true_max_locations["Levy"]),
    # (Powell(dim=10).to(device, dtype=dtype), torch.tensor([[-4., -4., -4., -4., -4., -4., -4., -4., -4., -4.], [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]], dtype=dtype).to(device), true_maxima["Powell"], true_max_locations["Powell"]),
    (Shekel().to(device, dtype=dtype), torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]], dtype=dtype).to(device), true_maxima["Shekel"], true_max_locations["Shekel"])
    # (StyblinskiTang(dim=6).to(device, dtype=dtype), torch.tensor([[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]], dtype=dtype).to(device), true_maxima["StyblinskiTang"], true_max_locations["StyblinskiTang"])
]

mean_results = []
titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]

for test_function, bounds, true_max, true_max_location in test_functions:
    print(f"Running experiments for {test_function.__class__.__name__}")

    mean_results1, _ = run_experiment(test_function, n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], n_runs, seeds, bounds)
    mean_results2, _ = run_experiment(test_function, n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'], n_runs, seeds, bounds)
    mean_results3, _ = run_experiment(test_function, n_iterations, ['Matern'], ['EI', 'UCB', 'PI'], n_runs, seeds, bounds)

    mean_results.append([mean_results1, mean_results2, mean_results3])

    plot_save_path = f"{test_function.__class__.__name__}_mean_performance_comparison.png"
    plot_results([mean_results1, mean_results2, mean_results3], titles, test_function.__class__.__name__, plot_save_path, true_max, true_max_location)
