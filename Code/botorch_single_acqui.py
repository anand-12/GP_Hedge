import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
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

# Define the EI acquisition function
def get_ei_acquisition_function(model, train_y):
    return ExpectedImprovement(model=model, best_f=train_y.min())

# Optimization loop
num_iterations = 20

for iteration in range(num_iterations):
    model, mll = initialize_model(train_x, train_y)
    fit_gpytorch_model(mll)

    ei_acqf = get_ei_acquisition_function(model, train_y)

    candidate, acq_value = optimize_acqf(
        acq_function=ei_acqf,
        bounds=torch.stack([torch.zeros(2, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype)]),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    new_x = candidate.detach()
    new_y = branin(new_x).unsqueeze(-1)
    
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

# Find the minimum value and the corresponding location
min_value = train_y.min().item()
min_index = train_y.argmin().item()
min_location = train_x[min_index]

print("Optimization finished.")
print("Best found value:", min_value)
print("Best found location:", min_location.cpu().numpy())

# Plot the Branin function
def plot_branin():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([branin(torch.tensor([[xx, yy]], device=device, dtype=dtype)).item() for xx, yy in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), c='red', s=20, label='Evaluated Points')
    plt.scatter(min_location.cpu()[0], min_location.cpu()[1], c='white', s=50, edgecolors='black', label='Best Point')
    plt.title('Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

plot_branin()
