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
n_iterations = 100
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

bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])  

def random_get_next_points(init_x, init_y, best_init_y, bounds, n_points=1, weights=None):
    single_model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)


    EI = ExpectedImprovement(model=single_model, best_f=best_init_y)
    UCB = UpperConfidenceBound(model=single_model, beta=0.1)
    PI = ProbabilityOfImprovement(model=single_model, best_f=best_init_y)
    
    acquisition_functions = [EI, UCB, PI]
    
    if weights is None:
        weights = np.ones(len(acquisition_functions)) / len(acquisition_functions)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=weights)
    chosen_acq_function = acquisition_functions[chosen_acq_index]

    candidates, _ = optimize_acqf(acq_function=chosen_acq_function, bounds=bounds, q=n_points, num_restarts=20, raw_samples=512, options={"batch_limit": 5, "maxiter": 200})

    return candidates, chosen_acq_index

def get_next_points(init_x, init_y, best_init_y, bounds, n_points=1, gains=None, eta=0.1):
    single_model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=single_model, best_f=best_init_y)
    UCB = UpperConfidenceBound(model=single_model, beta=0.1)
    PI = ProbabilityOfImprovement(model=single_model, best_f=best_init_y)
    
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

    return candidates_list[chosen_acq_index], chosen_acq_index, single_model

def run_optimization(acquisition_function, n_iterations=n_iterations):
    init_x, init_y, best_init_y = generate_initial_data(5)
    best_observed_values = []

    for i in range(n_iterations):
        print(f"Number of iterations done: {i}")
        single_model = SingleTaskGP(init_x, init_y)
        mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_mll(mll)

        if acquisition_function == UpperConfidenceBound:
            acq_func = acquisition_function(model=single_model, beta=0.1)
        else:
            acq_func = acquisition_function(model=single_model, best_f=best_init_y)
        
        candidates, _ = optimize_acqf(acq_function=acq_func, 
                                      bounds=bounds, q=1, num_restarts=20, raw_samples=512, options={"batch_limit": 5, "maxiter": 200})
        new_results = target_function(candidates).unsqueeze(-1)

        init_x = torch.cat([init_x, candidates])
        init_y = torch.cat([init_y, new_results])

        best_init_y = init_y.max().item()
        best_observed_values.append(best_init_y)

    return best_observed_values

# GP-Hedge optimization
init_x, init_y, best_init_y = generate_initial_data(5)
gains = np.zeros(3)
eta = 0.1
best_observed_values_gp_hedge = []
chosen_acq_functions = []


for i in range(n_iterations):
    print(f"Number of iterations done: {i}")
    new_candidates, chosen_acq_index, single_model = get_next_points(init_x, init_y, best_init_y, bounds, 1, gains, eta)
    new_results = target_function(new_candidates).unsqueeze(-1)

    print(f"New candidates are: {new_candidates}")
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
    best_observed_values_gp_hedge.append(best_init_y)
    chosen_acq_functions.append(chosen_acq_index)
    print(f"Best point performs this way: {best_init_y}")

    posterior_mean = single_model.posterior(new_candidates).mean
    reward = posterior_mean.mean().item()
    gains[chosen_acq_index] += reward

best_observed_values_ei = run_optimization(ExpectedImprovement)
print("EI done")
best_observed_values_ucb = run_optimization(UpperConfidenceBound)
print("UCB done")
best_observed_values_pi = run_optimization(ProbabilityOfImprovement)
print("PI done")

# Random portfolio optimization
weights = np.ones(3) / 3
best_observed_values_rp = []
init_x, init_y, best_init_y = generate_initial_data(20)
chosen_acq_functions_rp = []

for i in range(n_iterations):
    print(f"Number of iterations done: {i}")
    new_candidates, chosen_acq_index = random_get_next_points(init_x, init_y, best_init_y, bounds, 1, weights)
    new_results = target_function(new_candidates).unsqueeze(-1)

    print(f"New candidates are: {new_candidates}")
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
    best_observed_values_rp.append(best_init_y)
    chosen_acq_functions_rp.append(chosen_acq_index)
    print(f"Best point performs this way: {best_init_y}")

    if new_results.max().item() > best_observed_values_rp[-1]:
        weights[chosen_acq_index] += 1.0
    weights = weights / weights.sum() 

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(best_observed_values_gp_hedge, label='GP-Hedge', marker='.', linestyle='-')
plt.plot(best_observed_values_ei, label='EI', marker='.', linestyle='-')
plt.plot(best_observed_values_ucb, label='UCB', marker='.', linestyle='-')
plt.plot(best_observed_values_pi, label='PI', marker='.', linestyle='-')
plt.plot(best_observed_values_rp, label='Random Portfolio', marker='.', linestyle='-')
plt.title("Performance Comparison of GP-Hedge, EI, UCB, PI, and Random Portfolio")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Function Value")
plt.legend()
plt.grid(True)
plt.show()
