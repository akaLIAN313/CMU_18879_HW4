import numpy as np
import ray
import os
import time
import pickle
from tqdm import trange
from scipy.signal import find_peaks # Keep this if find_spike_times is not in helper
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
# Ensure these modules contain the necessary functions
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
from helper import calculate_fr, calculate_latency, calculate_cv
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from MBO_logger import log_experiment_results


start_time = time.time()
timestamp = time.strftime("%Y%m%d-%H%M%S")
ray.init(ignore_reinit_error=True, num_cpus=2) # Adjust num_cpus as needed

# --- Simulation & Optimization Parameters ---
total_time = 500 # ms
# Define parameter grid (adjust density as needed)
amp1_range = np.linspace(1, 200, 60, dtype=int)
amp2_range = np.linspace(1, 200, 60, dtype=int)
freq1_range = np.linspace(1, 100, 30, dtype=int)
freq2_range = np.linspace(1, 100, 30, dtype=int)
X1, X2, X3, X4 = np.meshgrid(amp1_range, amp2_range, freq1_range, freq2_range, indexing='ij')
grid_points = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()))
n_grid_points = len(grid_points)

init_trials = 5  # Number of initial random evaluations
opt_trials = 15 # Number of BO iterations
n_objective_functions = 6 # Updated number of objectives

# --- Objective Functions & Evaluation ---
def deterministic_power(x):
    """Calculates deterministic power based on amplitudes."""
    amp1, amp2, *_ = x
    return (amp1**2 + amp2**2) / 2 # Example power calculation

def simulate_and_evaluate(x):
    """Runs simulations and calculates all objectives."""
    amp1, amp2, freq1, freq2 = x
    results = ray.get([
        simulation_Pyr.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, total_time=total_time, plot_waveform=False),
        simulation_PV.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, total_time=total_time, plot_waveform=False)
    ])
    (response_Pyr, t), (response_PV, _) = results

    # Calculate objectives
    fr_Pyr = calculate_fr(response_Pyr, t)
    fr_PV = calculate_fr(response_PV, t)
    power = deterministic_power(x) # Deterministic objective
    latency = calculate_latency(response_PV, t) # Stochastic objective
    cv_Pyr = calculate_cv(response_Pyr, t) # New stochastic objective
    cv_PV = calculate_cv(response_PV, t)   # New stochastic objective

    # Handle potential NaN from CV (assign a high value for minimization)
    latency = latency if not np.isnan(latency) else t[-1] # Assign max time if NaN
    cv_Pyr = cv_Pyr if not np.isnan(cv_Pyr) else 10.0
    cv_PV = cv_PV if not np.isnan(cv_PV) else 10.0

    # Return all objectives in a defined order
    # Order: [fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV]
    return fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV

# --- GP Fitting and Sampling ---
def fit_gp_and_sample_function(X, y, n_features=500):
    """Fits GP using RFF approximation and returns a function sampling from it."""
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    D = X_scaled.shape[1]
    W = np.random.normal(0, 1, size=(n_features, D))
    b = np.random.uniform(0, 2 * np.pi, size=(n_features))
    Z = np.sqrt(2 / n_features) * np.cos(X_scaled @ W.T + b)
    gp = GaussianProcessRegressor(kernel=RBF(), alpha=1e-6, normalize_y=True)
    gp.fit(Z, y)
    def sampled_function(X_query):
        Xq = scaler.transform(X_query)
        Zq = np.sqrt(2 / n_features) * np.cos(Xq @ W.T + b)
        # Ensure gp.sample_y returns appropriate shape, flatten if needed
        samples = gp.sample_y(Zq, n_samples=1)
        return samples.flatten() if samples.ndim > 1 else samples
    return sampled_function

# --- MOBO Problem Definition for Pymoo ---
class GPProblem(Problem):
    """Pymoo Problem definition using sampled GP functions."""
    def __init__(self, f_pv_fr, f_pyr_fr, f_latency, f_pyr_cv, f_pv_cv):
        # n_obj is now 6
        super().__init__(n_var=4, n_obj=n_objective_functions,
                         xl=np.array([amp1_range.min(), amp2_range.min(), freq1_range.min(), freq2_range.min()]),
                         xu=np.array([amp1_range.max(), amp2_range.max(), freq1_range.max(), freq2_range.max()]))
        self.f_pv_fr = f_pv_fr     # Sampled func for PV FR
        self.f_pyr_fr = f_pyr_fr    # Sampled func for Pyr FR
        self.f_latency = f_latency # Sampled func for Latency
        self.f_pyr_cv = f_pyr_cv     # Sampled func for Pyr CV
        self.f_pv_cv = f_pv_cv      # Sampled func for PV CV

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate objectives using sampled functions or deterministic calculation
        # Negate objectives to be maximized (here: PV FR)
        f1_vals = -self.f_pv_fr(X)  # Max PV FR -> Min (-PV_FR)
        f2_vals = self.f_pyr_fr(X)  # Min Pyr FR
        f3_vals = np.array([deterministic_power(x) for x in X]) # Min Power
        f4_vals = self.f_latency(X) # Min Latency
        f5_vals = self.f_pyr_cv(X)  # Min Pyr CV
        f6_vals = self.f_pv_cv(X)   # Min PV CV
        out["F"] = np.column_stack([f1_vals, f2_vals, f3_vals, f4_vals, f5_vals, f6_vals])

# --- Pareto Front Sampling ---
def sample_pareto_frontiers(X_data, Y_data, n_samples=10):
    """Samples multiple Pareto fronts from the GP models."""
    fronts = []
    n_features_rff = 500 # Number of random features for GP approximation

    # Indices corresponding to simulate_and_evaluate return order
    idx = {'pyr_fr': 0, 'pv_fr': 1, 'power': 2, 'latency': 3, 'pyr_cv': 4, 'pv_cv': 5}

    for _ in range(n_samples):
        # Create sampled functions for each stochastic objective
        f_pv_fr_s = fit_gp_and_sample_function(X_data, Y_data[:, idx['pv_fr']], n_features=n_features_rff)
        f_pyr_fr_s = fit_gp_and_sample_function(X_data, Y_data[:, idx['pyr_fr']], n_features=n_features_rff)
        f_latency_s = fit_gp_and_sample_function(X_data, Y_data[:, idx['latency']], n_features=n_features_rff)
        f_pyr_cv_s = fit_gp_and_sample_function(X_data, Y_data[:, idx['pyr_cv']], n_features=n_features_rff)
        f_pv_cv_s = fit_gp_and_sample_function(X_data, Y_data[:, idx['pv_cv']], n_features=n_features_rff)

        # Define and solve the optimization problem for this sample
        problem = GPProblem(f_pv_fr_s, f_pyr_fr_s, f_latency_s, f_pyr_cv_s, f_pv_cv_s)
        algorithm = NSGA2(pop_size=100) # Adjust pop_size if needed for higher dimensions
        termination = get_termination("n_gen", 40) # Adjust generations if needed
        res = minimize(problem, algorithm, termination, seed=None, verbose=False)
        if res.F is not None and len(res.F) > 0:
             fronts.append(res.F)
        else:
             print("Warning: NSGA2 returned empty front for one sample.") # Handle cases where optimization fails

    return fronts

# --- Acquisition Function (PFES Approximation) ---
def approximate_pfes_acquisition(X_query, gps, sampled_fronts):
    """Calculates acquisition score based on non-domination probability w.r.t. sampled fronts."""
    # Predict means for all stochastic objectives using provided GPs dictionary
    mu_pyr_fr, _ = gps['pyr_fr'].predict(X_query, return_std=True)
    mu_pv_fr, _ = gps['pv_fr'].predict(X_query, return_std=True)
    mu_latency, _ = gps['latency'].predict(X_query, return_std=True)
    mu_pyr_cv, _ = gps['pyr_cv'].predict(X_query, return_std=True)
    mu_pv_cv, _ = gps['pv_cv'].predict(X_query, return_std=True)

    # Calculate deterministic objective mean
    mu_power = np.array([deterministic_power(x) for x in X_query])

    # Construct the mean objective vector in the Pymoo problem's format
    predicted_means = np.column_stack([
        -mu_pv_fr,   # Negated for maximization
        mu_pyr_fr,
        mu_power,
        mu_latency,
        mu_pyr_cv,
        mu_pv_cv
    ])

    scores = np.zeros(len(X_query))
    if not sampled_fronts: # Handle case with no valid sampled fronts
        print("Warning: No sampled fronts available for acquisition calculation.")
        return scores

    n_objectives = sampled_fronts[0].shape[1]
    for pf in sampled_fronts:
        # Check for each query point if it's dominated by ANY point in the current sampled front 'pf'
        # dominated[i] is True if query point i is dominated by at least one point in pf
        dominated = np.any(np.all(predicted_means[:, None, :] >= pf[None, :, :], axis=2), axis=1)
        # Increment score for non-dominated points
        scores += ~dominated

    return scores

# --- Initial Data Generation or Loading ---
def sample_initial_points(grid_pts, n_points):
    """Selects initial points randomly from the grid."""
    indices = np.random.choice(len(grid_pts), size=n_points, replace=False)
    return grid_pts[indices]

# Updated cache file name for the new objective set
cache_file = f"init_cache_{n_objective_functions}_obj.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        X_data, Y_data = pickle.load(f) # Load X and combined Y data
    print(f"Loaded cached initial evaluations ({Y_data.shape[1]} objectives).")
    # Optional: Validate shape
    if Y_data.shape[1] != n_objective_functions:
         print(f"Warning: Cache file has {Y_data.shape[1]} objectives, expected {n_objective_functions}. Re-running initialization.")
         os.remove(cache_file) # Invalidate cache
         X_data, Y_data = None, None # Reset data
else:
    X_data, Y_data = None, None

# init with cache
if X_data is None:
    X_init = sample_initial_points(grid_points, init_trials)
    Y_init = np.zeros((len(X_init), n_objective_functions))
    print("Running initial evaluations...")
    for i, x in enumerate(X_init):
        obj_vals = simulate_and_evaluate(x)
        print(f"Initial eval {i}: {np.round(obj_vals, 2)}")
        Y_init[i] = np.array(obj_vals, dtype=float)
    X_data = X_init.copy()
    Y_data = Y_init.copy()
    with open(cache_file, "wb") as f:
        pickle.dump((X_data, Y_data), f)
    print(f"Initial evaluations completed and cached ({n_objective_functions} objectives).")

print(f"\nStarting BO loop for {opt_trials} iterations...")
for i in trange(opt_trials, desc="BO Iteration", unit="iter"):
    gp_models = {} # Store GPs in a dictionary
    idx = {'pyr_fr': 0, 'pv_fr': 1, 'latency': 3, 'pyr_cv': 4, 'pv_cv': 5} # Exclude power (idx 2)
    for name, col_idx in idx.items():
        gp = GaussianProcessRegressor(RBF(), alpha=1e-6, normalize_y=True)
        gp.fit(X_data, Y_data[:, col_idx])
        gp_models[name] = gp

    try:
         sampled_fronts = sample_pareto_frontiers(X_data, Y_data)
    except Exception as e:
         print(f"\nError during Pareto front sampling: {e}")
         # Decide how to proceed: skip iteration, use previous fronts, etc.
         print("Skipping iteration due to sampling error.")
         continue # Skip to next iteration

    if not sampled_fronts:
        print("\nWarning: No valid Pareto fronts sampled. Skipping acquisition calculation.")
        next_idx = np.random.choice(len(grid_points))
        next_x = grid_points[next_idx]
        print("Picking random point as fallback.")
    else:
        acquisition_scores = approximate_pfes_acquisition(grid_points, gp_models, sampled_fronts)
        next_idx = np.argmax(acquisition_scores)
        next_x = grid_points[next_idx]
        #  Check if selected point is already evaluated
        if np.any(np.all(X_data == next_x, axis=1)):
             print(f"\nWarning: Point {next_x} already evaluated. Choosing point with second highest score.")
             sorted_indices = np.argsort(acquisition_scores)[::-1]
             for idx_choice in sorted_indices[1:]:
                 potential_next_x = grid_points[idx_choice]
                 if not np.any(np.all(X_data == potential_next_x, axis=1)):
                     next_x = potential_next_x
                     next_idx = idx_choice
                     print(f"Selected alternative point: {next_x}")
                     break
             else:
                  print("All high-scoring points evaluated, choosing random.")
                  available_indices = np.setdiff1d(np.arange(len(grid_points)), np.where(np.all(X_data == grid_points[:,None], axis=2))[0], assume_unique=True)
                  next_idx = np.random.choice(available_indices)
                  next_x = grid_points[next_idx]

    obj_vals = simulate_and_evaluate(next_x)
    print(f"\nIter {i}: Selected {next_x}, Objectives: {np.round(obj_vals, 2)}")

    X_data = np.vstack([X_data, next_x])
    Y_data = np.vstack([Y_data, np.array(obj_vals)])

end_time = time.time()
duration = end_time - start_time
print(f"\nTotal runtime: {duration:.2f} sec ({duration/60:.2f} min)")
try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = 'interactive_or_unknown'
print("Logging experiment results...")
log_experiment_results(
    X_data=X_data,
    Y_data=Y_data,
    script_name=script_name,
    duration=duration,
)

print("Sampling final Pareto frontier...")
try:
    final_fronts = sample_pareto_frontiers(X_data, Y_data, n_samples=20)  # Increase samples for final frontier
    if final_fronts:
        # Save all sampled fronts
        final_frontier_file = f"pareto_fronts_{script_name}_{timestamp}.npz"
        np.savez(
            final_frontier_file,
            fronts=np.array(final_fronts),
            objectives=['PV_FR (max)', 'Pyr_FR (min)', 'Power (min)', 
                       'Latency (min)', 'Pyr_CV (min)', 'PV_CV (min)']
        )
        print(f"Saved {len(final_fronts)} Pareto fronts to {final_frontier_file}")
    else:
        print("Warning: Failed to generate final Pareto fronts")
except Exception as e:
    print(f"Error sampling final Pareto fronts: {e}")

print("Script finished.")
ray.shutdown()
