import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import glob
import re
import argparse

def create_visualization(points, ax, plot_type='both', alpha=0.3):
    """Create visualization based on specified type."""
    if plot_type in ['surface', 'both']:
        try:
            tri = Delaunay(points[:, :2])
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                          triangles=tri.simplices,
                          alpha=alpha,
                          cmap='viridis',
                          edgecolor='none')
        except Exception as e:
            print(f"Warning: Could not create surface, falling back to scatter.")
            plot_type = 'scatter'
            
    if plot_type in ['scatter', 'both']:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='black', s=10, alpha=0.6)

def extract_solution(log_path):
    """Extract the best solution from a log file."""
    with open(log_path, 'r') as f:
        for line in f:
            if 'Current best solution:' in line:
                match = re.search(r'objectives=\[(.*?)\]', line)
                if match:
                    return np.array([float(x) for x in match.group(1).split()])
    return None

def main():
    parser = argparse.ArgumentParser(description='Plot Pareto fronts with optimization solutions')
    parser.add_argument('--plot-type', choices=['surface', 'scatter', 'both'], 
                       default='both', help='Type of plot to generate')
    parser.add_argument('--n-fronts', type=int, default=None,
                       help='Number of Pareto fronts to plot (default: all)')
    args = parser.parse_args()

    # Load data
    data = np.load('pareto_fronts_PFES.py_20250424-102310.npz', allow_pickle=True)
    fronts = data['fronts'][:args.n_fronts] if args.n_fronts else data['fronts']
    objectives = data['objectives']

    # Load solutions
    bo_log = glob.glob('logs/BO_6.py_*.log')[-1]
    mba_log = glob.glob('logs/MBA.py_*.log')[-1]
    
    bo_solution = extract_solution(bo_log)
    mba_solution = extract_solution(mba_log)

    # Setup plotting
    obj_indices = list(combinations(range(len(objectives)), 3))
    n_combinations = len(obj_indices)
    n_rows = (n_combinations + 3) // 4
    n_cols = min(4, n_combinations)

    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    plot_type_str = 'Surface and Scatter' if args.plot_type == 'both' else args.plot_type.capitalize()
    plt.suptitle(f'Pareto Fronts ({plot_type_str}) with Optimization Solutions\n'
                 f'Using {len(fronts)} frontiers', fontsize=16, y=0.95)

    for idx, (i, j, k) in enumerate(obj_indices, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
        
        # Plot Pareto front
        all_points = np.vstack([front[:, [i, j, k]] for front in fronts])
        create_visualization(all_points, ax, args.plot_type)
        
        # Plot solutions
        if bo_solution is not None:
            ax.scatter(bo_solution[i], bo_solution[j], bo_solution[k],
                      c='red', s=100, marker='*', label='BO Solution')
        if mba_solution is not None:
            ax.scatter(mba_solution[i], mba_solution[j], mba_solution[k],
                      c='blue', s=100, marker='^', label='MBA Solution')
        
        # Labels
        ax.set_xlabel(objectives[i].split()[0])
        ax.set_ylabel(objectives[j].split()[0])
        ax.set_zlabel(objectives[k].split()[0])
        
        title = f'{objectives[i].split()[0]} vs\n{objectives[j].split()[0]} vs\n{objectives[k].split()[0]}'
        ax.set_title(title, fontsize=8)
        
        if idx == 1:
            ax.legend()
        
        ax.view_init(elev=20, azim=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = f'figs/pareto_{args.plot_type}_{len(fronts)}fronts_with_solutions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_path}")
    
    # Print solutions
    print("\nBO Solution:")
    for obj_name, value in zip(objectives, bo_solution):
        print(f"{obj_name}: {value:.4f}")
        
    print("\nMBA Solution:")
    for obj_name, value in zip(objectives, mba_solution):
        print(f"{obj_name}: {value:.4f}")

if __name__ == "__main__":
    main()