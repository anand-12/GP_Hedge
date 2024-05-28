import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.test_functions import Branin

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Define the objective function (Branin)
branin = Branin().to(dtype=dtype, device=device)

# Generate initial data
def generate_initial_data(n=5):
    train_x = torch.rand(n, 2, device=device, dtype=dtype)
    train_y = branin(train_x).unsqueeze(-1)
    return train_x, train_y

train_x, train_y = generate_initial_data()

# Define the GP model
def initialize_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

model, mll = initialize_model(train_x, train_y)

# Define the acquisition functions
def get_acquisition_functions(model):
    EI = ExpectedImprovement(model=model, best_f=train_y.min())
    PI = ProbabilityOfImprovement(model=model, best_f=train_y.min())
    # UCB = UpperConfidenceBound(model=model, beta=0.1)
    return [EI, PI]

acquisition_functions = get_acquisition_functions(model)

# Define the GP-Hedge mechanism
def select_acquisition_function(acquisition_functions, weights):
    with torch.no_grad():
        scores = torch.tensor([acqf(train_x.unsqueeze(1)).mean().item() for acqf in acquisition_functions])
    probs = torch.softmax(scores * weights, dim=0)
    selected_acqf_idx = torch.multinomial(probs, 1).item()
    return acquisition_functions[selected_acqf_idx], selected_acqf_idx

# Optimization loop
num_iterations = 20
weights = torch.ones(len(acquisition_functions))

for iteration in range(num_iterations):
    model, mll = initialize_model(train_x, train_y)
    fit_gpytorch_model(mll)

    acquisition_functions = get_acquisition_functions(model)
    selected_acqf, selected_acqf_idx = select_acquisition_function(acquisition_functions, weights)

    candidate, acq_value = optimize_acqf(
        acq_function=selected_acqf,
        bounds=torch.stack([torch.zeros(2, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype)]),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    new_x = candidate.detach()
    new_y = branin(new_x).unsqueeze(-1)
    
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    # Update weights based on the improvement achieved
    improvement = train_y.min() - selected_acqf.best_f
    weights[selected_acqf_idx] *= (1 + improvement).clamp_min(1.0)

print("Optimization finished.")
print("Best found value:", train_y.min().item())
