import numpy as np
import ray
import os
import time
import pickle
from tqdm import trange
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
from helper import calculate_fr, calculate_latency, calculate_cv
from MBO_logger import log_experiment_results

# Initialize Ray for parallel processing
start_time = time.time()
ray.init(ignore_reinit_error=True, num_cpus=2)

# Define parameter space and bounds
total_time = 500
param_bounds = {
    'amp1': (1, 200),
    'amp2': (1, 200),
    'freq1': (1, 100),
    'freq2': (1, 100)
}

# Define step sizes for each parameter (the "jump" size)
step_sizes = {
    'amp1': 10,   # Increase/decrease by 10 units
    'amp2': 10,   # Increase/decrease by 10 units
    'freq1': 5,   # Increase/decrease by 5 units
    'freq2': 5    # Increase/decrease by 5 units
}

# Define actions (arms) - each arm is a directional movement in parameter space
actions = [
    ('amp1', 'increase'),
    ('amp1', 'decrease'),
    ('amp2', 'increase'),
    ('amp2', 'decrease'),
    ('freq1', 'increase'),
    ('freq1', 'decrease'),
    ('freq2', 'increase'),
    ('freq2', 'decrease'),
    ('no_change', 'no_change')  # Stay at current position (useful for exploitation)
]

# Configuration parameters
init_trials = 5   # Initial random exploration
mab_trials = 15   # MAB iterations

# Define objective weights
weights = {
    'PV': 0.4,       # Maximize PV firing rate
    'Pyr': -0.3,     # Minimize Pyr firing rate
    'power': -0.005, # Minimize power
    'latency': -0.1, # Minimize latency
    'pyr_cv': -1,    # Minimize Pyr CV
    'pv_cv': -1      # Minimize PV CV
}

def deterministic_power(x):
    """Calculate power consumption based on stimulation amplitudes"""
    amp1, amp2, *_ = x
    return (amp1**2 + amp2**2) / 2

def sample_random_point():
    """Sample a random point within the parameter bounds"""
    amp1 = np.random.randint(param_bounds['amp1'][0], param_bounds['amp1'][1] + 1)
    amp2 = np.random.randint(param_bounds['amp2'][0], param_bounds['amp2'][1] + 1)
    freq1 = np.random.randint(param_bounds['freq1'][0], param_bounds['freq1'][1] + 1)
    freq2 = np.random.randint(param_bounds['freq2'][0], param_bounds['freq2'][1] + 1)
    return np.array([amp1, amp2, freq1, freq2])

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

class DirectionalMAB:
    """
    Directional Multi-Armed Bandit for parameter optimization
    Each arm is a directional movement (increase/decrease) of a specific parameter
    """
    
    def __init__(self, actions, epsilon=0.1, alpha=0.1, optimistic_init=True):
        """
        Initialize DirectionalMAB
        
        Parameters:
        -----------
        actions : list
            List of action tuples (parameter, direction)
        epsilon : float
            Exploration probability for epsilon-greedy strategy
        alpha : float
            Learning rate for reward updates
        optimistic_init : bool
            Whether to use optimistic initialization (to encourage exploration)
        """
        self.actions = actions
        self.num_arms = len(actions)
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Initialize Q-values (expected rewards) for each arm
        if optimistic_init:
            self.q_values = np.ones(self.num_arms) * 5.0  # Optimistic initialization
        else:
            self.q_values = np.zeros(self.num_arms)
            
        self.counts = np.zeros(self.num_arms)
    
    def select_arm(self):
        """Select arm using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Exploration: pick a random arm
            return np.random.randint(self.num_arms)
        else:
            # Exploitation: pick the best arm
            return np.argmax(self.q_values)
    
    def update(self, arm_idx, reward):
        """Update Q-value estimate for the pulled arm"""
        self.counts[arm_idx] += 1
        
        # Update Q-value using running average
        self.q_values[arm_idx] += self.alpha * (reward - self.q_values[arm_idx])

def apply_action(current_params, action):
    """
    Apply the selected action (arm) to the current parameters.
    Returns new parameter values after applying the action.
    """
    param_name, direction = action
    new_params = current_params.copy()
    
    if param_name == 'no_change':
        return new_params
    
    param_idx = {'amp1': 0, 'amp2': 1, 'freq1': 2, 'freq2': 3}[param_name]
    step = step_sizes[param_name]
    
    if direction == 'increase':
        new_params[param_idx] = min(new_params[param_idx] + step, param_bounds[param_name][1])
    else:  # decrease
        new_params[param_idx] = max(new_params[param_idx] - step, param_bounds[param_name][0])
    
    return new_params

# Load or create initial data
cache_file = "init_cache_directional_mab.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        X_data, Y_data, weighted_data = pickle.load(f)
    print("Loaded cached initial evaluations.")
else:
    # Sample initial random points
    X_init = np.array([sample_random_point() for _ in range(init_trials)])
    Y_init = np.zeros((len(X_init), 6))  # 6 objectives
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

# Initialize the Directional MAB
mab = DirectionalMAB(actions, epsilon=0.2, alpha=0.1, optimistic_init=True)

# Start from the best point found so far
best_idx = np.argmax(weighted_data)
current_params = X_data[best_idx].copy()
current_reward = weighted_data[best_idx]

print(f"Starting MAB optimization from best initial point: {current_params}")
print(f"Initial weighted reward: {current_reward}")

# Main MAB optimization loop
for i in trange(mab_trials, desc="MAB Iteration", unit="iter"):
    # Select action (arm)
    arm_idx = mab.select_arm()
    action = actions[arm_idx]
    
    # Apply the selected action to get new parameters
    new_params = apply_action(current_params, action)
    
    # Check if parameters changed (could be unchanged if at boundary)
    params_changed = not np.array_equal(new_params, current_params)
    
    # Only evaluate if parameters actually changed
    if params_changed:
        # Evaluate the new parameters
        fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV, weighted_sum = simulate_and_evaluate(new_params)
        
        # Update data storage
        X_data = np.vstack([X_data, new_params])
        Y_data = np.vstack([Y_data, [fr_Pyr, fr_PV, power, latency, cv_Pyr, cv_PV]])
        weighted_data = np.append(weighted_data, weighted_sum)
        
        # Calculate reward as improvement over current position
        reward = weighted_sum - current_reward
        
        print(f"Iter {i}: Action={action}, New params={new_params}, "
              f"Reward={reward}, Weighted={weighted_sum}")
        
        # Update MAB with reward
        mab.update(arm_idx, reward)
        
        # If new position is better, move there
        if weighted_sum > current_reward:
            current_params = new_params
            current_reward = weighted_sum
            print(f"  Moved to new position with better reward: {current_reward}")
    else:
        # If parameters didn't change (hit boundary), give negative reward
        mab.update(arm_idx, -1.0)
        print(f"Iter {i}: Action={action}, No change in parameters (boundary reached), "
              f"Negative reward applied")
    
    # Adaptive exploration: decrease epsilon over time
    if i > mab_trials // 2:
        mab.epsilon = max(0.05, mab.epsilon * 0.9)  # Reduce exploration
    
    # Report current best solution across all evaluations
    best_idx = np.argmax(weighted_data)
    best_param = X_data[best_idx]
    best_objectives = Y_data[best_idx]
    
    print(f"Current best overall: params={best_param}, weighted={weighted_data[best_idx]}")

# Find and report final best solution
best_idx = np.argmax(weighted_data)
best_param = X_data[best_idx]
best_objectives = Y_data[best_idx]

print("\nFinal Best Solution:")
print(f"Parameters: amp1={best_param[0]}, amp2={best_param[1]}, freq1={best_param[2]}, freq2={best_param[3]}")
print(f"Objectives: Pyr={best_objectives[0]}, PV={best_objectives[1]}, Power={best_objectives[2]}, "
      f"Latency={best_objectives[3]}, Pyr_CV={best_objectives[4]}, PV_CV={best_objectives[5]}")
print(f"Weighted Objective: {weighted_data[best_idx]}")

# Report best actions
print("\nAction Performance:")
for i, action in enumerate(actions):
    print(f"Action {action}: Q-value={mab.q_values[i]:.4f}, Times selected={mab.counts[i]}")

end_time = time.time()
duration = end_time - start_time
print(f"Total runtime: {duration:.2f} sec ({duration/60:.2f} min)")

try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = 'directional_mab_multi_objective'

# Log results
log_experiment_results(X_data, Y_data, script_name=script_name, duration=duration)

# Save final data
with open("directional_mab_results.pkl", "wb") as f:
    results_dict = {
        'X_data': X_data,
        'Y_data': Y_data,
        'weighted_data': weighted_data,
        'weights': weights,
        'best_solution': {
            'parameters': best_param,
            'objectives': best_objectives,
            'weighted_value': weighted_data[best_idx]
        },
        'mab_results': {
            'q_values': mab.q_values,
            'counts': mab.counts,
            'actions': actions
        }
    }
    pickle.dump(results_dict, f)