from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
from helper import firing_rate
import ray

# Initialize Ray with multiple workers
ray.init(ignore_reinit_error=True, num_cpus=2)

total_time = 500 #total simulation time in ms

# Below is an example of how to run the simulations in parallel for both neurons
# Define the parameters for the waveform
# These parameters need to be optimized by Bayesian optimization
amp1 = 200
amp2 = 100
freq1 = 5
freq2 = 100

results = [simulation_Pyr.remote(
            num_electrode = 1,  
            amp1 = amp1, amp2 = amp2, freq1 = freq1, freq2 = freq2,
            total_time = total_time,  
            plot_waveform = False # Set to True to plot injected current
        ),
        simulation_PV.remote(
            num_electrode=1,  
            amp1 = amp1, amp2 = amp2, freq1 = freq1, freq2 = freq2,
            total_time = total_time,  
            plot_waveform = False # Set to True to plot injected current
        )]

# Retrieve results from Ray
(response_Pyr, t), (response_PV, t) = ray.get(results)

# Compute firing rates
FR_Pyr = firing_rate(response_Pyr, 500)
FR_PV = firing_rate(response_PV, 500)

print("Firing rate of Pyramidal neuron: ", FR_Pyr)
print("Firing rate of PV neuron: ", FR_PV)
ray.shutdown()