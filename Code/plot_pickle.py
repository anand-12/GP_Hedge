import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def load_pickle_files(base_path, dataset_name, n_runs):
    all_runs = []
    for run in range(1, n_runs + 1):
        file_pattern = f"{dataset_name}_run_{run}_seed_"
        matching_files = [file for file in os.listdir(base_path) if file.startswith(file_pattern) and file.endswith('.pkl')]
        if not matching_files:
            print(f"No matching files for pattern: {file_pattern}")
            continue
        filepath = os.path.join(base_path, matching_files[0])
        with open(filepath, 'rb') as f:
            run_data = pickle.load(f)
        all_runs.append(run_data)
    return np.array(all_runs)

def plot_mean_performance(mean_results, titles):
    plt.figure(figsize=(12, 8))

    for mean_best_observed_values, title in zip(mean_results, titles):
        plt.plot(mean_best_observed_values, marker='o', linestyle='-', label=title)
    
    plt.title("Mean Performance Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

# Base path for the pickle files
base_path = "/export/home/anandr/GP_Hedge/Code"

# Dataset names
dataset_names = ["Hartmann"]

n_runs = 10
mean_results = []
save_path = "/export/home/anandr/GP_Hedge/Hartmann_performance_comparison.png"
# Load and aggregate results for each dataset
for dataset_name in dataset_names:
    print(f"Loading data for {dataset_name}")
    all_runs = load_pickle_files(base_path, dataset_name, n_runs)
    if all_runs.size > 0:
        mean_best_observed_values = np.mean(all_runs, axis=0)
        mean_results.append(mean_best_observed_values)
    else:
        print(f"No data found for {dataset_name}")

# Titles for the plots
titles = [
    "Hartmann Function"
]

# Plot the mean performance
plot_mean_performance(mean_results, titles)
