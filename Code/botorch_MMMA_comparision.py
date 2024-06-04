import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
import warnings

warnings.filterwarnings("ignore")

hart6 = Hartmann(dim=6)

def target_function(individuals):
    result = []
    for x in individuals:
        result.append(-1.0 * hart6(x))
    return torch.tensor(result)

def generate_initial_data(n=10):
    train_x = torch.rand(n, 6, dtype=torch.double)  
    exact_obj = target_function(train_x).unsqueeze(-1)
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

    model = CustomGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
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

def run_experiment(n_iterations, kernel_types, acq_func_types):
    init_x, init_y, best_init_y = generate_initial_data(20)
    bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])  
    gains = np.zeros(len(acq_func_types))
    eta = 0.1  

    best_observed_values = []
    chosen_acq_functions = []
    selected_models = []

    train_x, train_y = init_x, init_y
    best_init_y = train_y.max().item()

    for t in range(n_iterations):
        print(f"Number of iterations done: {t}")
        new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(train_x, train_y, best_init_y, bounds, eta, 1, gains, kernel_types, acq_func_types)
        new_results = target_function(new_candidates).unsqueeze(-1)

        train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

        best_init_y = train_y.max().item()
        best_observed_values.append(best_init_y)
        chosen_acq_functions.append(chosen_acq_index)
        selected_models.append(selected_model_index)
        print(f"Best point performs this way: {best_init_y}")

        posterior_mean = selected_model.posterior(new_candidates).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return best_observed_values, chosen_acq_functions, selected_models

def plot_results(results, titles):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    for i, (best_observed_values, chosen_acq_functions, selected_models) in enumerate(results):
        axs[i].plot(best_observed_values, marker='o', linestyle='-', color='b')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Best Objective Function Value")
        axs[i].grid(True)
        
        ax2 = axs[i].twinx()
        ax2.plot(chosen_acq_functions, marker='o', linestyle='-', color='r')
        ax2.set_ylabel("Acquisition Function Index")
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["EI", "UCB", "PI"])
        ax2.grid(True)

        ax3 = axs[i].twinx()
        ax3.plot(selected_models, marker='o', linestyle='-', color='g')
        ax3.set_ylabel("Model Index")
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["RBF", "Matern", "RQ"])
        ax3.spines["right"].set_position(("outward", 60))
        ax3.grid(True)

    plt.tight_layout()
    plt.show()

n_iterations = 100

results1 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'])

results2 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'])

results3 = run_experiment(n_iterations, ['Matern'], ['EI', 'UCB', 'PI'])

results = [results1, results2, results3]
titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]

plot_results(results, titles)
