import numpy as np
import ray
import os
import time
import pickle
from tqdm import trange
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
from helper import calculate_fr, calculate_latency, calculate_cv
from MBO_logger import log_experiment_results

# Initialize Ray for parallel processing
start_time = time.time()
ray.init(ignore_reinit_error=True, num_cpus=2)

# Define parameter space
total_time = 500
amp1_range = np.linspace(1, 200, 60, dtype=int)
amp2_range = np.linspace(1, 200, 60, dtype=int)
freq1_range = np.linspace(1, 100, 30, dtype=int)
freq2_range = np.linspace(1, 100, 30, dtype=int)
X1, X2, X3, X4 = np.meshgrid(amp1_range, amp2_range, freq1_range, freq2_range, indexing='ij')
grid_points = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()))
n_grid_points = len(grid_points)

# Configuration parameters
init_trials = 5  # Initial random samples
opt_trials = 15   # BO iterations

# Define objective weights
# Adjust these weights based on the relative importance of each objective
# Positive weight means we want to maximize, negative means minimize
weights = {
    'PV': 0.4,      # We want to maximize PV firing rate (positive weight)
    'Pyr': -0.3,    # We want to minimize Pyr firing rate (negative weight)
    'power': -0.005, # We want to minimize power (negative weight)
    'latency': -0.1, # We want to minimize latency (negative weight)
    'pyr_cv': -1, # We want to minimize Pyr CV (negative weight)
    'pv_cv': -1   # We want to minimize PV CV (negative weight)
}

def deterministic_power(x):
    """Calculate power consumption based on stimulation amplitudes"""
    amp1, amp2, *_ = x
    return (amp1**2 + amp2**2) / 2

def sample_initial_points():
    """Randomly sample initial points from the grid"""
    return grid_points[np.random.randint(0, n_grid_points, size=(init_trials))]

def simulate_and_evaluate(x):
    """Run simulations and calculate all objectives including CV"""
    amp1, amp2, freq1, freq2 = x
    results = ray.get([
        simulation_Pyr.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, 
                             total_time=total_time, plot_waveform=False),
        simulation_PV.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, 
                           total_time=total_time, plot_waveform=False)
    ])
    (response_Pyr, t), (response_PV, _) = results
    
    fr_Pyr = calculate_fr(response_Pyr, t)
    fr_PV = calculate_fr(response_PV, t)
    power = deterministic_power(x)
    latency = calculate_latency(response_PV, t)
    cv_Pyr = calculate_cv(response_Pyr, t)
    cv_PV = calculate_cv(response_PV, t)

    # Handle NaN values
    if np.isnan(latency):
        latency = t[-1]
    if np.isnan(cv_Pyr):
        cv_Pyr = 10.0
    if np.isnan(cv_PV):
        cv_PV = 10.0
    
    # Calculate weighted sum of objectives
    weighted_sum = (
        weights['PV'] * fr_PV + 
        weights['Pyr'] * fr_Pyr + 
        weights['power'] * power + 
        weights['latency'] * latency +
        weights['pyr_cv'] * cv_Pyr +
        weights['pv_cv'] * cv_PV
    )
    
    return fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV, weighted_sum

def expected_improvement(X, gp, y_best):
    """Calculate expected improvement acquisition function"""
    mu, sigma = gp.predict(X, return_std=True)
    # We're maximizing, so improvement is when we exceed current best
    imp = mu - y_best
    # Handle case where sigma is zero
    mask = sigma > 0
    Z = np.zeros_like(sigma)
    Z[mask] = imp[mask] / sigma[mask]
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    # If sigma is zero, we have no uncertainty and can't improve
    ei[~mask] = 0.0
    return ei

# Load or create initial data
cache_file = "init_cache_6_obj.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        X_data, Y_data = pickle.load(f)
        weighted_init = np.zeros(len(X_data))
        for i, x in enumerate(Y_data):
            weighted_init[i] = (
                weights['PV'] * x[1] +
                weights['Pyr'] * x[0] +
                weights['power'] * x[2] +
                weights['latency'] * x[3] +
                weights['pyr_cv'] * x[4] +
                weights['pv_cv'] * x[5]
            )
        weighted_data = weighted_init.copy()
    print("Loaded cached initial evaluations.")
else:
    X_init = sample_initial_points()
    Y_init = np.zeros((len(X_init), 6))  # Updated to store 6 objectives
    weighted_init = np.zeros(len(X_init))
    
    for i, x in enumerate(X_init):
        fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV, weighted_sum = simulate_and_evaluate(x)
        Y_init[i] = np.array([fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV], dtype=float)
        weighted_init[i] = weighted_sum
        print(f"Initial {i}: Pyr {fr_Pyr}, PV {fr_PV}, Power {power}, Latency {latency}, "
              f"Pyr_CV {cv_Pyr}, PV_CV {cv_PV}, Weighted {weighted_sum}")

    X_data = X_init.copy()
    Y_data = Y_init.copy()
    weighted_data = weighted_init.copy()

    with open(cache_file, "wb") as f:
        pickle.dump((X_data, Y_data, weighted_data), f)
    print("Initial evaluations completed and cached.")

# Import needed only for acquisition function
from scipy.stats import norm

# Main Bayesian Optimization loop
for i in trange(opt_trials, desc="BO Iteration", unit="iter"):
    # Define kernel and fit GP
    # Kernel choice affects performance - Matern often works well for physical processes
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
    
    # Scale features for better GP performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Fit GP to weighted objective
    gp.fit(X_scaled, weighted_data)
    
    # Scale grid points for prediction
    grid_scaled = scaler.transform(grid_points)
    
    # Find current best value (we're maximizing the weighted sum)
    best_idx = np.argmax(weighted_data)
    best_value = weighted_data[best_idx]
    
    # Calculate acquisition function (Expected Improvement)
    ei_values = expected_improvement(grid_scaled, gp, best_value)
    
    # Select next point to evaluate
    next_idx = np.argmax(ei_values)
    next_x = grid_points[next_idx]
    
    # Evaluate new point
    fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV, weighted_sum = simulate_and_evaluate(next_x)
    
    # Update data
    X_data = np.vstack([X_data, next_x])
    Y_data = np.vstack([Y_data, [fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV]])
    weighted_data = np.append(weighted_data, weighted_sum)
    
    print(f"Iter {i}: Pyr {fr_Pyr}, PV {fr_PV}, Power {power}, Latency {latency}, Pyr_CV {cv_Pyr}, PV_CV {cv_PV}, Weighted {weighted_sum}")
    
    # Store best solution found so far
    best_idx = np.argmax(weighted_data)
    best_param = X_data[best_idx]
    best_objectives = Y_data[best_idx]
    
    print(f"Current best solution: parameters={best_param}, objectives={best_objectives}, weighted={weighted_data[best_idx]}")

# Find and report final best solution
best_idx = np.argmax(weighted_data)
best_param = X_data[best_idx]
best_objectives = Y_data[best_idx]

print("\nFinal Best Solution:")
print(f"Parameters: amp1={best_param[0]}, amp2={best_param[1]}, freq1={best_param[2]}, freq2={best_param[3]}")
print(f"Objectives: Pyr={best_objectives[0]}, PV={best_objectives[1]}, Power={best_objectives[2]}, "
      f"Latency={best_objectives[3]}, Pyr_CV={best_objectives[4]}, PV_CV={best_objectives[5]}")
print(f"Weighted Objective: {weighted_data[best_idx]}")

end_time = time.time()
duration = end_time - start_time
print(f"Total runtime: {duration:.2f} sec ({duration/60:.2f} min)")

try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = 'weighted_bo'

# Log results (using the same logger as the original code)
log_experiment_results(X_data, Y_data, script_name=script_name, duration=duration)

# Save final data
with open("weighted_bo_6_obj_results.pkl", "wb") as f:
    results_dict = {
        'X_data': X_data,
        'Y_data': Y_data,
        'weighted_data': weighted_data,
        'weights': weights,
        'best_solution': {
            'parameters': best_param,
            'objectives': best_objectives,
            'weighted_value': weighted_data[best_idx]
        }
    }
    pickle.dump(results_dict, f)