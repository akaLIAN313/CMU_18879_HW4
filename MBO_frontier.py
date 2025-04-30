import numpy as np
import ray
import os
import time
import pickle
from tqdm import trange
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
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
ray.init(ignore_reinit_error=True, num_cpus=2)

total_time = 500
amp1_range = np.linspace(1, 200, 60, dtype=int)
amp2_range = np.linspace(1, 200, 60, dtype=int)
freq1_range = np.linspace(1, 100, 30, dtype=int)
freq2_range = np.linspace(1, 100, 30, dtype=int)
X1, X2, X3, X4 = np.meshgrid(amp1_range, amp2_range, freq1_range, freq2_range, indexing='ij')
grid_points = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()))
n_grid_points = len(grid_points)

init_trials = 5
opt_trials = 15

def deterministic_power(x):
    amp1, amp2, *_ = x
    return (amp1**2 + amp2**2) / 2

def sample_initial_points():
    return grid_points[np.random.randint(0, n_grid_points, size=(init_trials))]

def fit_gp_and_sample_function(X, y, n_features=500):
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
        return gp.sample_y(Zq, n_samples=1).flatten()
    return sampled_function

def simulate_and_evaluate(x):
    amp1, amp2, freq1, freq2 = x
    results = ray.get([
        simulation_Pyr.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, total_time=total_time, plot_waveform=False),
        simulation_PV.remote(num_electrode=1, amp1=amp1, amp2=amp2, freq1=freq1, freq2=freq2, total_time=total_time, plot_waveform=False)
    ])
    (response_Pyr, t), (response_PV, _) = results
    
    fr_Pyr = calculate_fr(response_Pyr, t)
    fr_PV = calculate_fr(response_PV, t)
    latency = calculate_latency(response_PV, t)
    power = deterministic_power(x)
    return fr_Pyr, fr_PV, power, latency

class GPProblem(Problem):
    def __init__(self, f1, f2, f4):
        super().__init__(n_var=4, n_obj=4, xl=1, xu=200)
        self.f1 = f1
        self.f2 = f2
        self.f4 = f4
    def _evaluate(self, X, out, *args, **kwargs):
        f1_vals = -self.f1(X)  # maximize PV
        f2_vals = self.f2(X)   # minimize Pyr
        f3_vals = np.array([deterministic_power(x) for x in X])  # deterministic power
        f4_vals = self.f4(X)  # response latency
        out["F"] = np.column_stack([f1_vals, f2_vals, f3_vals, f4_vals])

def sample_pareto_frontiers(X_data, y_data, latency_data, n_samples=10):
    fronts = []
    for _ in range(n_samples):
        f1 = fit_gp_and_sample_function(X_data, y_data[:, 1])  # PV
        f2 = fit_gp_and_sample_function(X_data, y_data[:, 0])  # Pyr
        f4 = fit_gp_and_sample_function(X_data, latency_data)  # latency
        problem = GPProblem(f1, f2, f4)
        algorithm = NSGA2(pop_size=100)
        res = minimize(problem, algorithm, get_termination("n_gen", 30), seed=1, verbose=False)
        fronts.append(res.F)
    return fronts

def approximate_pfes_acquisition(X_query, gp_pyr, gp_pv, gp_latency, sampled_fronts):
    mu_pyr, _ = gp_pyr.predict(X_query, return_std=True)
    mu_pv, _ = gp_pv.predict(X_query, return_std=True)
    mu_latency, _ = gp_latency.predict(X_query, return_std=True)
    mu_power = np.array([deterministic_power(x) for x in X_query])
    scores = np.zeros(len(X_query))
    for pf in sampled_fronts:
        dominated = (
            (mu_pyr[:, None] >= pf[:, 1]) &
            (mu_pv[:, None] <= pf[:, 0]) &
            (mu_power[:, None] <= pf[:, 2]) &
            (mu_latency[:, None] <= pf[:, 3])
        ).any(axis=1)
        scores += ~dominated
    return scores

cache_file = "init_cache.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        X_data, y_data, power_data, latency_data = pickle.load(f)
    print("Loaded cached initial evaluations.")
else:
    X_init = sample_initial_points()
    Y_init = np.zeros((len(X_init), 4))
    for i, x in enumerate(X_init):
        obj_vals = simulate_and_evaluate(x)
        print(obj_vals)
        Y_init[i] = np.array(obj_vals, dtype=float)

    X_data = X_init.copy()
    y_data = Y_init[:, :2]
    power_data = Y_init[:, 2]
    latency_data = Y_init[:, 3]

    with open(cache_file, "wb") as f:
        pickle.dump((X_data, y_data, power_data, latency_data), f)
    print("Initial evaluations completed and cached.")

for i in trange(opt_trials, desc="BO Iteration", unit="iter"):
    gp_pyr = GaussianProcessRegressor(RBF(), alpha=1e-6, normalize_y=True)
    gp_pv = GaussianProcessRegressor(RBF(), alpha=1e-6, normalize_y=True)
    gp_latency = GaussianProcessRegressor(RBF(), alpha=1e-6, normalize_y=True)
    gp_pyr.fit(X_data, y_data[:, 0])
    gp_pv.fit(X_data, y_data[:, 1])
    gp_latency.fit(X_data, latency_data)

    sampled_fronts = sample_pareto_frontiers(X_data, y_data, latency_data)
    acquisition_scores = approximate_pfes_acquisition(grid_points, gp_pyr, gp_pv, gp_latency, sampled_fronts)
    next_x = grid_points[np.argmax(acquisition_scores)]

    fr_Pyr, fr_PV, power, latency = simulate_and_evaluate(next_x)
    X_data = np.vstack([X_data, next_x])
    y_data = np.vstack([y_data, [fr_Pyr, fr_PV]])
    power_data = np.append(power_data, power)
    latency_data = np.append(latency_data, latency)
    print(f"Iter {i}: Pyr {fr_Pyr}, PV {fr_PV}, Power {power}, Latency {latency}")

end_time = time.time()
duration = end_time - start_time
print(f"Total runtime: {duration:.2f} sec ({duration/60:.2f} min)")

try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = 'interactive_or_unknown'

log_experiment_results(X_data, y_data, power_data, script_name=script_name, duration=duration)

