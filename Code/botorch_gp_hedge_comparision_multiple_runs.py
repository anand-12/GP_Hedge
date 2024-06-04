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
import time

warnings.filterwarnings("ignore")

hart6 = Hartmann(dim=6)
n_iterations = 100
n_seeds = 5

def target_function(individuals):
    result = []
    for x in individuals:
        result.append(-1.0 * hart6(x))
    return torch.tensor(result)

def generate_initial_data(n=5):
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

def run_optimization(acquisition_function, seed, n_iterations=n_iterations):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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

def run_gp_hedge(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    init_x, init_y, best_init_y = generate_initial_data(5)
    gains = np.zeros(3)
    eta = 0.1
    best_observed_values = []
    chosen_acq_functions = []

    for i in range(n_iterations):
        print(f"Number of iterations done: {i}")
        new_candidates, chosen_acq_index, single_model = get_next_points(init_x, init_y, best_init_y, bounds, 1, gains, eta)
        new_results = target_function(new_candidates).unsqueeze(-1)

        print(f"New candidates are: {new_candidates}")
        init_x = torch.cat([init_x, new_candidates])
        init_y = torch.cat([init_y, new_results])

        best_init_y = init_y.max().item()
        best_observed_values.append(best_init_y)
        chosen_acq_functions.append(chosen_acq_index)
        print(f"Best point performs this way: {best_init_y}")

        posterior_mean = single_model.posterior(new_candidates).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return best_observed_values, chosen_acq_functions

def run_random_portfolio(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    weights = np.ones(3) / 3
    best_observed_values = []
    init_x, init_y, best_init_y = generate_initial_data(20)
    chosen_acq_functions = []

    for i in range(n_iterations):
        print(f"Number of iterations done: {i}")
        new_candidates, chosen_acq_index = random_get_next_points(init_x, init_y, best_init_y, bounds, 1, weights)
        new_results = target_function(new_candidates).unsqueeze(-1)

        print(f"New candidates are: {new_candidates}")
        init_x = torch.cat([init_x, new_candidates])
        init_y = torch.cat([init_y, new_results])

        best_init_y = init_y.max().item()
        best_observed_values.append(best_init_y)
        chosen_acq_functions.append(chosen_acq_index)
        print(f"Best point performs this way: {best_init_y}")

        if new_results.max().item() > best_observed_values[-1]:
            weights[chosen_acq_index] += 1.0
        weights = weights / weights.sum()

    return best_observed_values

def plot_chosen_acq_functions(chosen_acq_functions, seed):
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_acq_functions, marker='o', linestyle='-', color='r')
    plt.title(f"Acquisition Function Selected at Each Iteration for Seed {seed}")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition Function Index")
    plt.yticks([0, 1, 2], ["EI", "UCB", "PI"])
    plt.grid(True)
    plt.show()

all_gp_hedge_results = []
all_ei_results = []
all_ucb_results = []
all_pi_results = []
all_rp_results = []

gp_hedge_times = []
ei_times = []
ucb_times = []
pi_times = []
rp_times = []

all_chosen_acq_functions = []

for seed in range(n_seeds):
    print(f"Running experiments with seed {seed}")

    start_time = time.time()
    print("Running GP-Hedge")
    gp_hedge_results, chosen_acq_functions = run_gp_hedge(seed)
    all_gp_hedge_results.append(gp_hedge_results)
    all_chosen_acq_functions.append(chosen_acq_functions)
    gp_hedge_times.append(time.time() - start_time)
    
    start_time = time.time()
    print("Running EI")
    all_ei_results.append(run_optimization(ExpectedImprovement, seed))
    ei_times.append(time.time() - start_time)
    
    start_time = time.time()
    print("Running UCB")
    all_ucb_results.append(run_optimization(UpperConfidenceBound, seed))
    ucb_times.append(time.time() - start_time)
    
    start_time = time.time()
    print("Running PI")
    all_pi_results.append(run_optimization(ProbabilityOfImprovement, seed))
    pi_times.append(time.time() - start_time)
    
    start_time = time.time()
    print("Running Random Portfolio")
    all_rp_results.append(run_random_portfolio(seed))
    rp_times.append(time.time() - start_time)

# Convert results to numpy arrays
all_gp_hedge_results = np.array(all_gp_hedge_results)
all_ei_results = np.array(all_ei_results)
all_ucb_results = np.array(all_ucb_results)
all_pi_results = np.array(all_pi_results)
all_rp_results = np.array(all_rp_results)

# Compute mean performance
mean_gp_hedge = np.mean(all_gp_hedge_results, axis=0)
mean_ei = np.mean(all_ei_results, axis=0)
mean_ucb = np.mean(all_ucb_results, axis=0)
mean_pi = np.mean(all_pi_results, axis=0)
mean_rp = np.mean(all_rp_results, axis=0)

# Compute mean time taken
mean_gp_hedge_time = np.mean(gp_hedge_times)
mean_ei_time = np.mean(ei_times)
mean_ucb_time = np.mean(ucb_times)
mean_pi_time = np.mean(pi_times)
mean_rp_time = np.mean(rp_times)

print(f"Average time taken for GP-Hedge: {mean_gp_hedge_time:.2f} seconds")
print(f"Average time taken for EI: {mean_ei_time:.2f} seconds")
print(f"Average time taken for UCB: {mean_ucb_time:.2f} seconds")
print(f"Average time taken for PI: {mean_pi_time:.2f} seconds")
print(f"Average time taken for Random Portfolio: {mean_rp_time:.2f} seconds")

# Plotting
plt.figure(figsize=(10, 6))
for result in all_gp_hedge_results:
    plt.plot(result, color='red', alpha=0.1)
plt.plot(mean_gp_hedge, label='GP-Hedge', marker='.', linestyle='-', color='red')

for result in all_ei_results:
    plt.plot(result, color='blue', alpha=0.1)
plt.plot(mean_ei, label='EI', marker='.', linestyle='-', color='blue')

for result in all_ucb_results:
    plt.plot(result, color='green', alpha=0.1)
plt.plot(mean_ucb, label='UCB', marker='.', linestyle='-', color='green')

for result in all_pi_results:
    plt.plot(result, color='purple', alpha=0.1)
plt.plot(mean_pi, label='PI', marker='.', linestyle='-', color='purple')

for result in all_rp_results:
    plt.plot(result, color='orange', alpha=0.1)
plt.plot(mean_rp, label='Random Portfolio', marker='.', linestyle='-', color='orange')

plt.title("Mean Performance Comparison of GP-Hedge, EI, UCB, PI, and Random Portfolio")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Function Value")
plt.legend()
plt.grid(True)
plt.show()

# Plot chosen acquisition functions for each seed in GP-Hedge
for seed, chosen_acq_functions in enumerate(all_chosen_acq_functions):
    plot_chosen_acq_functions(chosen_acq_functions, seed)
