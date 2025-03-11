import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler  
from numpy.random import seed

#Ground truth objective function
def objective(x): 
    return -(np.sin(x) + 0.1 * x ** 2)

def sample_initial_points():
    #TODO: Randomly sample 10 initial data points in the range of [-10, 10]
    pass

def acquisition_function():
    #TODO: Implement the Expected Improvement (EI) acquisition function
    pass

def BO_loop():
    #TODO: Implement the Bayesian Optimization loop
    opt_trials = 10 
    for i in range(opt_trials):
        pass

# Define the GP model
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, optimizer=None) 



# TODO: Plot the following:
# - Ground truth objective function
# - GP predicted function
# - Sampled data points 
