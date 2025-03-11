import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
import math
from helper import fibonacci_sphere

### Plot Pyr Neurons
cell_id_pyr_lst = [6,7,8,9,10]
temp=34
dt=0.025
human_or_mice = 1
loc_pyr = np.array([0,0,0])
angle_pyr= np.array([0,0])
coord_lst = []
for cell_id_pyr in cell_id_pyr_lst:
    ## Get Neuron Coordinates
    neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt)
    coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pyr, angle=angle_pyr))
    coord_lst.append(coord.copy())

displace_lst = [np.array([0,-1500,2000]), np.array([0,1500,2000]),np.array([0,0,0]),np.array([0,2000,0]),np.array([0,-2000,0])]
for coord, displace in zip(coord_lst, displace_lst):
    coord[:,:] = coord[:,:]+displace

SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/Neuron_Plots')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

savepath = os.path.join(SAVE_PATH,'Pyr_Morphology')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for coord in coord_lst:
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0)
ax.set_xlabel('X-axis (um)', fontsize=14)
ax.set_ylabel('Y-axis (um)', fontsize=14)
ax.set_zlabel('Z-axis (um)', fontsize=14)
ax.set_title('Morphology of Pyr Neuron Models ', fontsize=21)
ax.tick_params(axis='x',labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(10,120)
plt.savefig(savepath+'_orientation1.png')
ax.view_init(10,240)
plt.savefig(savepath+'_orientation2.png')
ax.view_init(10,90)
plt.savefig(savepath+'_orientation3.png')
ax.view_init(10,0)
plt.savefig(savepath+'_orientation4.png')
view_angle = np.linspace(0,360,361)
def update(frame):
    ax.view_init(10,view_angle[frame])
ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
ani.save(os.path.join(savepath+'.gif'), writer='pillow')
ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
plt.show()
    
### Plot PV Neurons
cell_id_pv_lst = [32,33,34,35,36]
temp=34
dt=0.025
human_or_mice = 1
loc_pv = np.array([0,0,0])
angle_pv= np.array([0,0])
coord_lst = []
for cell_id_pv in cell_id_pv_lst:
    ## Get Neuron Coordinates
    neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv, temp=temp, dt=dt)
    coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pv, angle=angle_pv))
    coord_lst.append(coord.copy())
    del neuron
displace_lst = [np.array([0,-1500,2000]), np.array([0,1500,2000]),np.array([0,0,0]),np.array([0,2000,0]),np.array([0,-2000,0])]
for coord, displace in zip(coord_lst, displace_lst):
    coord[:,:] = coord[:,:]+displace/2
savepath = os.path.join(SAVE_PATH,'PV_Morphology')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for coord in coord_lst:
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0)
ax.set_xlabel('X-axis (um)', fontsize=14)
ax.set_ylabel('Y-axis (um)', fontsize=14)
ax.set_zlabel('Z-axis (um)', fontsize=14)
ax.set_title('Morphology of PV Neuron Models', fontsize=21)
ax.tick_params(axis='x',labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(10,120)
plt.savefig(savepath+'_orientation1.png')
ax.view_init(10,240)
plt.savefig(savepath+'_orientation2.png')
ax.view_init(10,90)
plt.savefig(savepath+'_orientation3.png')
ax.view_init(10,0)
plt.savefig(savepath+'_orientation4.png')
view_angle = np.linspace(0,360,361)
def update(frame):
    ax.view_init(10,view_angle[frame])
ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
ani.save(os.path.join(savepath+'.gif'), writer='pillow')
ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
plt.show()


