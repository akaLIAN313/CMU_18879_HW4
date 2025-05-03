# timulate Smart, Not Hard: A Pareto-Optimal Approach to Adaptive Neural Optimization
Lian Ren, Fuming Yang
CMU 18879 Final Project Code Iplementation
This repo is based on HW code from CMU 18-879 course.
## Set up
This repo could be set up using the method in origin README. Or by running `./setup_uv.sh` by using uv to install required packages faster.
## Implemented Methods
Three methods are implemted here to solve the multi-objective optimization problem, defined in [our presentation](https://docs.google.com/presentation/d/1w-WLmTy92AR_1F1v7GCPvPC6dnTAEs7u/edit?usp=sharing&ouid=105195536225465761374&rtpof=true&sd=true). Just run the corresding python scripts to see the results.
### PFES
in `PFES.py`
### Bayesian Optimization
in `BO_6.py`
### Multi-arm Bandit
in `MAB.py`
## Comparison
###
To plot the Pareto frontier proveded by PFES, run `plot_pareto_fronts_with_fronts_with_BO_MAB.py`
###
To compare the performance between methods towards linear transformation of multiple objects as evaluation metrix, run `compare_weighted_objects.py`

## origin README
For running the code: 
1. Run the following command in the terminal: sh init_setup_run_once.sh
2. Run starter.py to perform a simulation and verify that everything is working correctly.
3. The first time you run a neuron simulation, the program may take longer to generate the initial save state. Subsequent simulations will run faster.
Common Issues: 
1. To run the simulations you need the NEURON software. Installing the NEURON software can be tricky. Please refer to their setup guide: https://www.neuron.yale.edu/neuron/download, in case init_setup_run_once.sh is not able to install neuron using pip
