import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann, Ackley
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
import warnings
import random

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hart6 = Hartmann(dim=6).to(device)
ackley2 = Ackley(dim=2).to(device)

# Define the true maxima for the test functions
true_maxima = {
    "Hartmann": 3.32237,  # For the 6-dimensional Hartmann function
    "Ackley": 0.0,  # For the 2-dimensional Ackley function
}

def target_function(individuals, test_function):
    result = []
    for x in individuals:
        result.append(-1.0 * test_function(x))
    return torch.tensor(result, dtype=torch.double).to(device)

def generate_initial_data(n, n_dim, test_function):
    train_x = torch.rand(n, n_dim, dtype=torch.double).to(device)
    exact_obj = target_function(train_x, test_function).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

def fit_model(train_x, train_y, kernel_type):
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

    model = CustomGP(train_x, train_y).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll

def calculate_weights(models, mlls, train_x, train_y):
    log_likelihoods = np.array([mll(models[i](train_x), train_y).sum().item() for i, mll in enumerate(mlls)])
    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood  
    weights = np.exp(log_likelihoods) / np.sum(np.exp(log_likelihoods))
    return weights

def select_model(weights):
    return np.random.choice(len(weights), p=weights)

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=['RBF', 'Matern', 'RQ'], acq_func_types=['EI', 'UCB', 'PI']):

    models = []
    mlls = []
    for kernel in kernel_types:
        model, mll = fit_model(train_x, train_y, kernel)
        models.append(model)
        mlls.append(mll)

    weights = calculate_weights(models, mlls, train_x, train_y)
    selected_model_index = select_model(weights)
    selected_model = models[selected_model_index]

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

    return candidates_list[chosen_acq_index], chosen_acq_index, selected_model_index, selected_model

def update_data(train_x, train_y, new_x, new_y):
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    return train_x, train_y

def run_experiment(n_iterations, kernel_types, acq_func_types, initial_data, seeds):
    all_best_observed_values = []
    
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        train_x, train_y, best_init_y = initial_data["train_x"], initial_data["train_y"], initial_data["best_init_y"]
        bounds = initial_data["bounds"]
        test_function = initial_data["test_function"]

        gains = np.zeros(len(acq_func_types))
        eta = 0.1  

        best_observed_values = []
        best_init_y = train_y.max().item()

        for t in range(n_iterations):
            print(f"Seed {seed}, Iteration {t+1}/{n_iterations}")
            new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(train_x, train_y, best_init_y, bounds, eta, 1, gains, kernel_types, acq_func_types)
            new_results = target_function(new_candidates, test_function).unsqueeze(-1)

            train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

            best_init_y = train_y.max().item()
            best_observed_values.append(best_init_y)

            posterior_mean = selected_model.posterior(new_candidates).mean
            reward = posterior_mean.mean().item()
            gains[chosen_acq_index] += reward

        all_best_observed_values.append(best_observed_values)
    
    mean_best_observed_values = np.mean(all_best_observed_values, axis=0)
    return mean_best_observed_values, all_best_observed_values

def plot_results(mean_results, all_results, titles, test_function_name, true_maximum):
    plt.figure(figsize=(12, 8))

    for mean_best_observed_values, title in zip(mean_results, titles):
        plt.plot(mean_best_observed_values, marker='.', linestyle='-', label=title)
    
    for results in all_results:
        for run in results:
            plt.plot(run, color='gray', alpha=0.3)

    plt.axhline(y=true_maximum, color='k', linestyle='--', label='True Maxima')
    plt.title(f"Mean Performance Comparison for {test_function_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results_{test_function_name}.png")
    plt.show()

n_iterations = 80
n_initial_points = 20
n_runs = 10
seeds = [i for i in range(n_runs)]

# Ackley function settings
ackley_bounds = torch.tensor([[-32.768, -32.768], [32.768, 32.768]]).to(device)
init_x, init_y, best_init_y = generate_initial_data(n_initial_points, n_dim=ackley_bounds.size(1), test_function=ackley2)
ackley_initial_data = {
    "train_x": init_x,
    "train_y": init_y,
    "best_init_y": best_init_y,
    "bounds": ackley_bounds,
    "test_function": ackley2,
    "true_maximum": true_maxima['Ackley']
}

# Hartmann function settings
hart6_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).to(device)
init_x, init_y, best_init_y = generate_initial_data(n_initial_points, n_dim=hart6_bounds.size(1), test_function=hart6)
hartmann_initial_data = {
    "train_x": init_x,
    "train_y": init_y,
    "best_init_y": best_init_y,
    "bounds": hart6_bounds,
    "test_function": hart6,
    "true_maximum": true_maxima['Hartmann']
}

# Run experiments for Ackley function
# print("Running experiments for Ackley function")
# ackley_results1, ackley_all_results1 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], ackley_initial_data, seeds)
# ackley_results2, ackley_all_results2 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'], ackley_initial_data, seeds)
# ackley_results3, ackley_all_results3 = run_experiment(n_iterations, ['Matern'], ['EI', 'UCB', 'PI'], ackley_initial_data, seeds)

# ackley_mean_results = [ackley_results1, ackley_results2, ackley_results3]
# ackley_all_results = [ackley_all_results1, ackley_all_results2, ackley_all_results3]
# ackley_titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]
# plot_results(ackley_mean_results, ackley_all_results, ackley_titles, "Ackley", true_maxima['Ackley'])

# Run experiments for Hartmann function
print("Running experiments for Hartmann function")
hartmann_results1, hartmann_all_results1 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], hartmann_initial_data, seeds)
hartmann_results2, hartmann_all_results2 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'], hartmann_initial_data, seeds)
hartmann_results3, hartmann_all_results3 = run_experiment(n_iterations, ['Matern'], ['EI', 'UCB', 'PI'], hartmann_initial_data, seeds)

hartmann_mean_results = [hartmann_results1, hartmann_results2, hartmann_results3]
hartmann_all_results = [hartmann_all_results1, hartmann_all_results2, hartmann_all_results3]
hartmann_titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]
plot_results(hartmann_mean_results, hartmann_all_results, hartmann_titles, "Hartmann", true_maxima['Hartmann'])
