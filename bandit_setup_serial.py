from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation
from helper import firing_rate
import numpy as np
import random 
import ray


ray.init()
#define a class
def running_average(rewards_table):
    running_avg = np.zeros(len(rewards_table))
    running_avg[0] = rewards_table[0]
    for i in range(1,len(rewards_table)):
        running_avg[i] = (running_avg[i-1]*i + rewards_table[i])/(i+1)
    return running_avg

@ray.remote
class Arm:
    def __init__(self, function_type, noise_variance,cell_ID,non_stationary,switch_points):
        self.transfer_function = function_type
        self.noise_variance = noise_variance
        self.cell_ID = cell_ID
        self.non_stationary = non_stationary
        self.switch_points = switch_points
        self.FR = -1
       
    
    def get_reward(self,stim_amp, stim_pulse_width,tp):
        #stim is the input to the transfer function
        #output is the reward
        if self.FR == -1:
            mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
            FR = firing_rate(mem_potential[0], t)
            self.FR = FR
        # apply transfer function to mem_potential
        # FR (neural response) at a deep brain target, modulated by the transfer function
        if self.non_stationary == True and tp in self.switch_points:
            print('firing rate: ',self.FR)
            self.FR *= np.random.uniform(0.1, 1.5)
            print('Non-stationary reward at time point %d for cell: %d changed!!!!!'%(tp,self.cell_ID))
            print('firing rate after change: ',self.FR)
        FR_deep = self.apply_transfer_function(self.FR)
        noisy_FR_deep = FR_deep + FR_deep*np.random.normal(0,self.noise_variance)
        return noisy_FR_deep
    
    def get_ground_truth_reward(self, stim_amp, stim_pulse_width, tp):
        if self.FR == -1:
            mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time=100, cell_type=self.cell_ID, plot_neuron_with_electrodes=False)
            FR = firing_rate(mem_potential[0], t)
            self.FR = FR

        return self.apply_transfer_function(self.FR)

    def apply_transfer_function(self,FR):
        if self.transfer_function == 'square':
            return FR**2
        elif self.transfer_function == 'linear':
            return FR * 1
        elif self.transfer_function == 'linear_half':
            return FR * 0.5
        elif self.transfer_function == 'log_plus_linear':
            return np.log(FR) + FR -1
        elif self.transfer_function == 'sqrt':
            return np.sqrt(FR)
        else:
            raise ValueError('Transfer function not recognized')

    
def epsilon_greedy_bandit(arms,epsilon):
    #init reward on each arm
    #arms is a list of tuples, each tuple contains pulse_width and amplitude
    total_rewards_table = np.zeros(len(arms))
    trials_table = np.zeros(len(arms))
    avg_rewards_table = np.zeros(len(arms))
    num_trials = 100
    reward_over_time = []
    action_over_time = []
    best_rewards_over_time = []
    best_actions_over_time = []
    for tp in range(num_trials):
        random_val = np.random.rand()
        
        if random_val < epsilon:
            #explore
            action = np.random.randint(len(arms))
            action_over_time.append(action)
        else:
            #exploit
            action = np.argmax(avg_rewards_table)
            action_over_time.append(action)
        #run simulation
        fr_reward = ray.get(arms[action].get_reward.remote(200, 50,tp))

        ground_truth_rewards = ray.get([arm.get_ground_truth_reward.remote(200, 50, tp) for arm in arms])
        best_rewards_over_time.append(max(ground_truth_rewards))
        best_actions_over_time.append(np.argmax(ground_truth_rewards))
        # Update rewards and averages
        trials_table[action] += 1
        total_rewards_table[action] += fr_reward
        avg_rewards_table[action] = total_rewards_table[action] / trials_table[action]
        reward_over_time.append(fr_reward)

    return avg_rewards_table, reward_over_time, action_over_time, best_rewards_over_time, best_actions_over_time

# Instantiate arms as Ray actors
arm1 = Arm.remote(function_type='linear_half', noise_variance=0.2, cell_ID=6,non_stationary = True,switch_points =[5,40,60,130,160])
arm2 = Arm.remote(function_type='log_plus_linear', noise_variance=0.1, cell_ID=7,non_stationary = True,switch_points =[5,40,60,130,160])
arm3 = Arm.remote(function_type='linear', noise_variance=0.1, cell_ID=35,non_stationary = True,switch_points = [5,40,60,130,160])
arm4 = Arm.remote(function_type='sqrt', noise_variance=0.2, cell_ID=36,non_stationary = True,switch_points  =[5,40,60,130,160])
arms = [arm1, arm2, arm3, arm4]

# Run epsilon-greedy
epsilon = 0.2  # Exploration rate
_, reward_over_time, action_over_time, best_rewards_over_time, best_actions_over_time = epsilon_greedy_bandit(arms, epsilon)

# Plot results
running_avg = running_average(reward_over_time)

plt.figure()
plt.plot(reward_over_time)
plt.title('Immediate Reward Over Time')
plt.show()

plt.figure()
plt.plot(running_avg,label = 'Running Average Reward')
plt.plot(best_rewards_over_time, label='Best Ground Truth Reward', color='r', linestyle='--')
plt.legend()
plt.show()

plt.figure()
plt.scatter(range(len(action_over_time)), action_over_time, marker='o', alpha=0.8)
plt.scatter(range(len(best_actions_over_time)), best_actions_over_time, marker='x', alpha=0.8)
plt.xlabel('Trial')
plt.ylabel('Action (Arm Index)')
plt.title('Chosen Action Over Time')
plt.show()

# Shutdown Ray
ray.shutdown()