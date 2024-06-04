import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
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

def get_next_points(train_x, train_y, best_init_y, bounds, n_points=1, gains=None):
    models = []
    mlls = []
    kernel_types = ['RBF', 'Matern', 'RQ']  
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

    acquisition_functions = [EI, UCB, PI]
    
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

def plot_best_observed(best_observed_values):
    plt.figure(figsize=(10, 6))
    plt.plot(best_observed_values, marker='o', linestyle='-', color='b')
    plt.title("Best Objective Function Value at Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.grid(True)
    plt.show()

def plot_chosen_acq_functions(chosen_acq_functions):
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_acq_functions, marker='o', linestyle='-', color='r')
    plt.title("Acquisition Function Selected at Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition Function Index")
    plt.yticks([0, 1, 2], ["EI", "UCB", "PI"])
    plt.grid(True)
    plt.show()

def plot_selected_models(selected_models):
    plt.figure(figsize=(10, 6))
    plt.plot(selected_models, marker='o', linestyle='-', color='g')
    plt.title("Model Selected at Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Model Index")
    plt.yticks([0, 1, 2], ["RBF", "Matern", "RQ"])
    plt.grid(True)
    plt.show()

init_x, init_y, best_init_y = generate_initial_data(20)
bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])  
gains = np.zeros(3)
eta = 0.1  

best_observed_values = []
chosen_acq_functions = []
selected_models = []

n_iterations = 100
train_x, train_y = init_x, init_y
best_init_y = train_y.max().item()

for t in range(n_iterations):
    print(f"Number of iterations done: {t}")
    new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(train_x, train_y, best_init_y, bounds, 1, gains)
    print(f"At iteration {t} the selected model is {selected_model} and the chosen acquisition function is {chosen_acq_index}")
    new_results = target_function(new_candidates).unsqueeze(-1)

    print(f"New candidates are: {new_candidates}")
    train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

    best_init_y = train_y.max().item()
    best_observed_values.append(best_init_y)
    chosen_acq_functions.append(chosen_acq_index)
    selected_models.append(selected_model_index)
    print(f"Best point performs this way: {best_init_y}")

    posterior_mean = selected_model.posterior(new_candidates).mean
    reward = posterior_mean.mean().item()
    gains[chosen_acq_index] += reward

print(f"Best observed result: {best_init_y}")
best_candidate = train_x[((train_y == best_init_y).nonzero(as_tuple=True)[0])][0][0]
print(f"Best location of observed result: {best_candidate}")

plot_best_observed(best_observed_values)
plot_chosen_acq_functions(chosen_acq_functions)
plot_selected_models(selected_models)
