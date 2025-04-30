import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
from scipy.signal import find_peaks
import numpy as np
import os

### Helper Functions
##################################################################################
##################################################################################
##################################################################################
def cart_to_sph(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
    theta = np.arcsin(pos[:,2].reshape(-1,1)/r).reshape(-1,1)
    phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
    sph_pos = np.hstack([r,theta,phi])
    return sph_pos
    
def sph_to_cart(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    x = pos[:,0]*np.cos(pos[:,1])*np.cos(pos[:,2])
    y = pos[:,0]*np.cos(pos[:,1])*np.sin(pos[:,2])
    z = pos[:,0]*np.sin(pos[:,1])
    cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
    return cart_pos

def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi*(math.sqrt(5.)-1.)  # golden angle in radians
    if samples == 1:
        return np.array([[0, 1, 0]]) 
    for i in range(samples):
        y = 1-(i/float(samples-1))*2  # y goes from 1 to -1
        radius = math.sqrt(1-y*y)  # radius at y
        theta = phi*i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        

    return np.array(points)
    
def plot_electrode_and_neuron(coord_elec, coord, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    img = ax.scatter(coord[:,0]*10**(-3),coord[:,1]*10**(-3),coord[:,2]*10**(-3), linewidth=1.0, s=2.0)
    img = ax.scatter(coord_elec[:,0]*10**(-3), coord_elec[:,1]*10**(-3), coord_elec[:,2]*10**(-3), linewidth=0.3, s=50)
    ax.set_xlabel('X-axis (mm)', fontsize=14)
    ax.set_ylabel('Y-axis (mm)', fontsize=14, labelpad=20)
    ax.set_zlabel('Z-axis (mm)', fontsize=14, labelpad=20)
    ax.set_title('Neuron Orientation w.r.t Electrode', fontsize=21)
    ax.tick_params(axis='x',which='both',labelsize=12)
    ax.tick_params(axis='y',which='both',labelsize=12, pad=10)
    ax.tick_params(axis='z',which='both',labelsize=12, pad=10)
    
    ax.view_init(25,120)
    plt.savefig(savepath+'_orientation1.png')
    ax.view_init(25,240)
    plt.savefig(savepath+'_orientation2.png')
    ax.view_init(25,90)
    plt.savefig(savepath+'_orientation3.png')
    ax.view_init(25,0)
    plt.savefig(savepath+'_orientation4.png')
  
    view_angle = np.linspace(0,360,361)
    def update(frame):
        ax.view_init(10,view_angle[frame])
    ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    #ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
    plt.show()

def sample_spherical(num_samples, theta_max, y_max, r=1):
    tot_samples = 0
    samples = []
    while tot_samples<num_samples:
        samples_cand = np.random.normal(loc=0, scale=1, size=(10000,3))
        samples_cand = samples_cand/np.sqrt(np.sum(samples_cand**2, axis=1)).reshape(-1,1)*r
        samples_cand = cart_to_sph(samples_cand)
        samples_cand = samples_cand[samples_cand[:,1]>theta_max]
        samples_cand = sph_to_cart(samples_cand)
        samples_cand = samples_cand[np.abs(samples_cand[:,1])<y_max]
        samples.append(samples_cand)
        tot_samples = tot_samples+samples_cand.shape[0]
    samples = np.vstack(samples)
    samples = samples[:num_samples]
    return samples

def sample_1d(num_samples, theta_max, r):
    samples_theta = np.pi/2-np.abs(np.linspace(-theta_max, theta_max, num_samples))
    samples_phi = np.hstack([0*np.ones(int(samples_theta.shape[0]/2)),np.pi*np.ones(int(samples_theta.shape[0]/2))]).reshape(-1,1)
    samples = np.hstack([r*np.ones([num_samples,1]),samples_theta.reshape(-1,1),samples_phi.reshape(-1,1)])
    samples = sph_to_cart(samples)
    return samples

def _find_spike_times(response, t, prominence=40, height=None, **kwargs):
    # Basic check for valid array inputs kept for core functionality
    response_arr= np.array(response, copy=True)  # ensures it's writable
    peak_idxs, _ = find_peaks(response_arr, prominence=prominence, height=height, **kwargs)
    if len(peak_idxs) == 0:
        return np.array([])
    else:
        # Ensure indices are within bounds of t
        peak_idxs = peak_idxs[peak_idxs < len(t)]
        return t[peak_idxs]

def calculate_fr(response, t, prominence=40, height=None, **kwargs):
    spike_times = _find_spike_times(response, t, prominence=prominence, height=height, **kwargs)
    n_spikes = len(spike_times)
    if n_spikes == 0: # Keep check for no spikes -> zero rate
        return 0.0
    duration_sec = (t[-1] - t[0]) / 1000.0 # Assumes len(t) > 1 and duration > 0
    firing_rate = n_spikes / duration_sec
    return firing_rate

def calculate_latency(response, t, prominence=40, height=None, **kwargs):
    spike_times = _find_spike_times(response, t, prominence=prominence, height=height, **kwargs)
    # Assumes spikes are found, otherwise index [0] will error
    if len(spike_times) == 0:
        return np.nan # Keep check for no spikes -> NaN latency
    return spike_times[0]

def calculate_cv(response, t, prominence=40, height=None, min_spikes_for_cv=3, **kwargs):
    spike_times = _find_spike_times(response, t, prominence=prominence, height=height, **kwargs)
    print(spike_times)
    n_spikes = len(spike_times)
    # Removed check for min_spikes_for_cv - will error if n_spikes < 2
    if n_spikes < 2: # Need at least 1 ISI
        return np.nan
    isis = np.diff(spike_times)
    # Removed check for len(isis) < 2
    if len(isis) == 0: # Need at least 1 ISI
        return np.nan
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    # Removed check for mean_isi > 1e-9 - division by zero possible
    # Handle potential division by zero explicitly
    if abs(mean_isi) < 1e-12: # Check against a very small number
        return np.nan
    cv = std_isi / mean_isi
    return cv


def sample_switch_points(total_iterations,num_switches = 5):
    switch_points = np.random.choice(np.arange(1,total_iterations),num_switches,replace=False)
    return switch_points
