import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
from elec_field import sparse_place_human
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI
import sys
import math

SEED = 1234
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/Non-invasive')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head
elec_field_id = int(sys.argv[1])
if elec_field_id == 0:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'C3-C4')

elif elec_field_id == 1:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'C3-Cz')
elif elec_field_id == 2:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'HD-TDCS')
else:
    raise Exception('Wrong Id Supplied for Electrode Configuration. Valid Options: 0->C3C4; 1->C3Cz; 2->HD-TDCS')

amp_level = np.load(os.path.join(SAVE_PATH,"Amplitude.npy"))
diff_TI_lst, diff_sin_lst = [], []
error_TI_lst, error_sin_lst = [], []
pyr_TI_lst, pyr_sin_lst = [], []
error_pyr_TI_lst, error_pyr_sin_lst = [], []
amp_lst = []
diff_TI_all_lst, diff_sin_all_lst = [], []
pyr_TI_all_lst, pyr_sin_all_lst = [], []

for l in range(len(amp_level)):

    round_start_time = time.time()

    ## Defining Saving Directories
    #########################################################################################

    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)

    ## Loading Data
    #######################################################################################
    start_time = time.time()
    print("Loading Raw Data for Amplitude Level %d..."%(l))
    fr_rate_sin_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_sin_fr.npy'))
    fr_rate_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'))
    fr_rate_sin_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_sin_fr.npy'))
    fr_rate_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'))
    print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

    idx_activ_pyr = fr_rate_ti_pyr>0
    if np.sum(idx_activ_pyr) > 0:
        diff_TI = fr_rate_ti_pv[idx_activ_pyr]-fr_rate_ti_pyr[idx_activ_pyr]
        diff_TI_all_lst.append(diff_TI)
        pyr_TI_all_lst.append(fr_rate_ti_pyr[idx_activ_pyr])

        diff_sin = fr_rate_sin_pv[idx_activ_pyr]-fr_rate_sin_pyr[idx_activ_pyr]
        diff_sin_all_lst.append(diff_sin)
        pyr_sin_all_lst.append(fr_rate_sin_pyr[idx_activ_pyr])

        amp_lst.append(amp_level[l])

labels = [str(int(amp_lst[i])) for i in range(len(amp_lst))]

#### Fig 5 h. when using elec_field_id == 0  for C3-C4 configurations
fig, ax = plt.subplots()
labels = [str(int(amp_lst[i])) for i in range(len(amp_lst))]
data = []
for i in range(len(labels)):
    data.append(diff_sin_all_lst[i])
    data.append(diff_TI_all_lst[i])
x_pos = np.array([0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22])
x_pos_label = np.array([0.5,3.5,6.5,9.5,12.5,15.5,18.5,21.5])
x_pos = x_pos[:2*len(labels)]
x_pos_label = x_pos_label[:len(labels)]
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
if elec_field_id ==0 or elec_field_id==1:
    plt.ylim(ymax=420)
else:
    plt.ylim(ymax=300)
plt.xlabel('Injected Current (mA)', fontsize=19)

plt.tight_layout()
if elec_field_id == 0:
    if not os.path.exists(os.path.join(cwd,'TISimResults/Figs5Main')):
        os.makedirs(os.path.join(cwd,'TISimResults/Figs5Main'))
    plt.savefig(os.path.join('TISimResults/Figs5Main',"Fig5-h.png"))
plt.savefig(os.path.join(SAVE_PATH,"PV-Pyr_FR_Diff_BP.png"))
plt.show()

#### Plotting Aggregrate Pyr Response in a Box Plot
fig, ax = plt.subplots()
labels = [str(int(amp_lst[i])) for i in range(len(amp_lst))]
data = []
for i in range(len(labels)):
    data.append(pyr_sin_all_lst[i])
    data.append(pyr_TI_all_lst[i])
x_pos = np.array([0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22])
x_pos_label = np.array([0.5,3.5,6.5,9.5,12.5,15.5,18.5,21.5])
x_pos = x_pos[:2*len(labels)]
x_pos_label = x_pos_label[:len(labels)]
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

plt.ylabel("Pyr Firing Rate (Hz)", fontsize=20)
plt.legend(fontsize=18, ncols=2)
plt.xticks(ticks=x_pos_label, labels=labels, fontsize=19)
plt.yticks(fontsize=19)
plt.xlabel('Injected Current (mA)', fontsize=19)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Pyr_FR_BP.png"))
plt.show()



