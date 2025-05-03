import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from itertools import product

def extract_solution(log_path):
    """Extract the best solution from a log file."""
    with open(log_path, 'r') as f:
        for line in f:
            if 'Current best solution:' in line:
                match = re.search(r'objectives=\[(.*?)\]', line)
                if match:
                    return np.array([float(x) for x in match.group(1).split()])
    return None

def calculate_weighted_objective(solution, weights):
    """Calculate weighted sum of objectives."""
    return np.sum(solution * weights)

def main():
    # Load Pareto fronts and solutions
    data = np.load('pareto_fronts_PFES.py_20250424-102310.npz', allow_pickle=True)
    fronts = data['fronts']
    objectives = data['objectives']
    
    bo_log = glob.glob('logs/BO_6.py_*.log')[-1]
    mba_log = glob.glob('logs/MBA.py_*.log')[-1]
    
    bo_solution = extract_solution(bo_log)
    mba_solution = extract_solution(mba_log)

    # Define weight combinations
    weight_scenarios = {
        'Balanced': np.array([0.2, -0.2, -0.15, -0.15, -0.15, -0.15]),
        'Performance': np.array([0.3, -0.3, -0.1, -0.2, -0.05, -0.05]),
        'Power_Efficient': np.array([0.15, -0.15, -0.4, -0.1, -0.1, -0.1]),
        'Stability': np.array([0.1, -0.1, -0.1, -0.1, -0.3, -0.3]),
    }

    # Calculate weighted objectives
    results = {}
    for scenario_name, weights in weight_scenarios.items():
        # Find best achievable value from Pareto front
        pareto_scores = []
        for front in fronts:
            front_scores = [calculate_weighted_objective(point, weights) for point in front]
            pareto_scores.extend(front_scores)
        
        bo_score = calculate_weighted_objective(bo_solution, weights)
        mba_score = calculate_weighted_objective(mba_solution, weights)
        
        results[scenario_name] = {
            'pareto_best': max(pareto_scores),  # Using max since we want best achievable
            'bo_score': bo_score,
            'mba_score': mba_score
        }

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar plot comparing solutions
    scenario_names = list(weight_scenarios.keys())
    x = np.arange(len(scenario_names))
    width = 0.25
    
    axes[0].bar(x - width, [results[s]['bo_score'] for s in scenario_names], 
                width, label='BO Solution', color='red', alpha=0.7)
    axes[0].bar(x, [results[s]['mba_score'] for s in scenario_names], 
                width, label='MBA Solution', color='blue', alpha=0.7)
    axes[0].bar(x + width, [results[s]['pareto_best'] for s in scenario_names], 
                width, label='Pareto Best', color='green', alpha=0.7)
    
    axes[0].set_ylabel('Weighted Objective Value')
    axes[0].set_title('Comparison with Best Achievable Pareto Solutions')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend()
    
    # Radar plot of weights
    angles = np.linspace(0, 2*np.pi, len(objectives), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(2, 1, 2, projection='polar')
    for scenario_name, weights in weight_scenarios.items():
        values = np.concatenate((weights, [weights[0]]))
        ax.plot(angles, values, label=scenario_name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([obj.split()[0] for obj in objectives])
    ax.set_title('Weight Distributions')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    save_path = 'figs/weighted_objective_comparison_best.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 50)
    for scenario, result in results.items():
        print(f"\n{scenario}:")
        print(f"  BO Score: {result['bo_score']:.4f}")
        print(f"  MBA Score: {result['mba_score']:.4f}")
        print(f"  Best Achievable (Pareto): {result['pareto_best']:.4f}")
        print(f"  Gap to Pareto:")
        print(f"    BO: {(result['pareto_best'] - result['bo_score']):.4f}")
        print(f"    MBA: {(result['pareto_best'] - result['mba_score']):.4f}")

    plt.show()

if __name__ == "__main__":
    main()