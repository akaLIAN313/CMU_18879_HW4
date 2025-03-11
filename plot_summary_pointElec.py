import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import ray
from neuron_model_parallel import NeuronSim
from helper import fibonacci_sphere, plot_electrode_and_neuron

LOAD_PATH = os.path.join(os.getcwd(),'TISimResults/PointElectrodeSim')
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/Figs5Main')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
filenames = ['Results_distance500.0um', 'Results_distance1000.0um', 'Results_distance2000.0um', 'Results_distance4000.0um', 'Results_distance8000.0um', 'Results_distance16000.0um', 'Results_uniform']
files_dir = [os.path.join(LOAD_PATH,file) for file in filenames if file[:6]=='Result']
activ_thresh_pyr, activ_thresh_pv  = [], []
fr_activ_thresh_pyr, fr_activ_thresh_pv  = [], []
labels = []
for file in files_dir:
    activ_thresh_pyr.append(np.load(os.path.join(file, 'activation_Pyr.npy')))
    activ_thresh_pv.append(np.load(os.path.join(file, 'activation_PV.npy')))
    fr_activ_thresh_pyr.append(np.load(os.path.join(file, 'activation_fr_Pyr.npy')))
    fr_activ_thresh_pv.append(np.load(os.path.join(file, 'activation_fr_PV.npy')))

activ_thresh_pyr, activ_thresh_pv = np.array(activ_thresh_pyr), np.array(activ_thresh_pv)
fr_activ_thresh_pyr, fr_activ_thresh_pv = np.array(fr_activ_thresh_pyr), np.array(fr_activ_thresh_pv)

#### Calculatin average statistics
activ_inc = (activ_thresh_pyr-activ_thresh_pv)/np.median(activ_thresh_pv,axis=1).reshape(-1,1)*100
percentage = np.sum(activ_inc[4:]<0)/(np.sum(activ_inc[4:]>=0)+np.sum(activ_inc[4:]<0))*100
print("The percentage of PV neurons having lower activation thresholda than Pyr neurons for transcranial fields: %.2f"%percentage)

#### Plotting Fig 5 d

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = np.median(activ_inc, axis=1) 
bar_container = ax.bar(labels, data)
ax.set_ylabel('% increase threshold', fontsize=19)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,'Fig5-d.png'))
plt.show()

#### Plotting Fig 5 c

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = activ_inc.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
plt.axhline(xmin=0, xmax=1,y=0, color='black', linestyle='--')

for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c='C0', alpha=0.4)
plt.ylabel("% increase threshold", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Fig5-c.png"))
plt.show()

#### Plotting Fig 5 b

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = []
print(fr_activ_thresh_pyr.shape)
for i in range(7):
    data.append(fr_activ_thresh_pyr[i,:,2]-fr_activ_thresh_pyr[i,:,0])
    data.append(fr_activ_thresh_pyr[i,:,3]-fr_activ_thresh_pyr[i,:,1])
x_pos = np.array([0,1,3,4,6,7,9,10,12,13,15,16,18,19])
x_pos_label = np.array([0.5,3.5,6.5,9.5,12.5,15.5,18.5])
x = []
for i in range(len(x_pos)):
    x.append(np.random.normal(x_pos[i], 0.04, data[i].shape[0]))
boxprops= dict(facecolor=(0,0,0,0), color='black', linewidth=2)
medianprops = dict(color='black', linewidth=2)
bp = plt.boxplot(data, patch_artist=True, positions=x_pos, boxprops=boxprops, medianprops=medianprops)
for i in range(len(x_pos)):
    if i%2==0:
        if i==0:
            plt.scatter(x[i], data[i].flatten(), c='C0', alpha=0.4, label='Pure')
        else:
            plt.scatter(x[i], data[i].flatten(), c='C0', alpha=0.4)
    else:
        if i==1:
            plt.scatter(x[i], data[i].flatten(), c='C1', alpha=0.4, label='Modulated')
        else:
            plt.scatter(x[i], data[i].flatten(), c='C1', alpha=0.4)
plt.ylabel("PV-Pyr\n Firing Rate (Hz)", fontsize=20)
plt.legend(fontsize=18, ncols=2)
plt.xticks(ticks=x_pos_label, labels=labels, fontsize=19)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Fig5-b.png"))
plt.show()


#### Plotting SI Fig 7 a

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = fr_activ_thresh_pyr[:,:,0]-fr_activ_thresh_pyr[:,:,1]
data = data.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
xlims = ax.get_xlim()
plt.hlines(0,xmin=xlims[0], xmax=xlims[1], linestyle='--', color='black')
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.ylabel("Pure - Mod. Sin Fr. Rate", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"SIFig7-a.png"))
plt.show()

#### Plotting SI Fig 7 b

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = fr_activ_thresh_pv[:,:,2]-fr_activ_thresh_pv[:,:,3]
data = data.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
xlims = ax.get_xlim()
plt.hlines(0,xmin=xlims[0], xmax=xlims[1], linestyle='--', color='black')
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
#plt.title('Diff. between PV Firing rates of TI \n and Sin at PV Activation Threshold', fontsize=22)
plt.ylabel("Pure - Mod. Sin Fr. Rate", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"SIFig7-b.png"))
plt.show()


#### PLot Fig 5-a
## Intialize Neuron Simulator
human_or_mice = ray.put(1) ## 0-> human and 1-> mice
cell_id_pyr = ray.put(10)
temp = ray.put(34.0) ## Celsius, temparature at which neurons are simulated
dt = ray.put(0.025) ## ms, discretization time step
angle_pyr = np.array([0,0]) ## parameter used for specifying rotation of Pyr morphology
loc_pyr = np.array([0,0,0]) ## parameter used for specifying location of Pyr morphology

neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt)
coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pyr, angle=angle_pyr))
elec_location_ICMS = fibonacci_sphere(samples=30) ## Sampling 30 approximately uniformly spaced electrode locations from a unit mm sphere

savepath_curr = os.path.join(SAVE_PATH,'Fig5-a.png')
#plot_electrode_and_neuron(coord_elec=elec_location_ICMS*10**3, coord=coord, savepath=savepath_curr)
