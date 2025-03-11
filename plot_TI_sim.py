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
from helper import cart_to_sph, sph_to_cart, sample_spherical

def plot_response(coord, values1, values2, y_displace, title=None, savepath=None, show=False, target=False):

    coord1, coord2 = coord.copy(), coord.copy()
    coord1[:,1] = coord1[:,1]-y_displace/2
    coord2[:,1] = coord2[:,1]+y_displace/2
    values = np.concatenate([values1,values2], axis=0)
    coord = np.concatenate([coord1.copy(),coord2.copy()], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    color_map = cm.ScalarMappable(cmap='viridis_r')
    alpha_scale = values.copy()
    start_transparency = 0.5
    alpha_scale = (alpha_scale-np.min(alpha_scale))/(np.max(alpha_scale)-np.min(alpha_scale))*(1-start_transparency)+start_transparency
    rgba = color_map.to_rgba(x=values, alpha=alpha_scale, norm=True)

    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2],c=rgba, linewidth=2.0)
    cbar=plt.colorbar(mappable=color_map, ax=ax, pad=-0.1, shrink=0.64)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xlabel('X-axis (cm)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y-axis (cm)', fontsize=20, labelpad=20)
    if title is not None:
        ax.set_title(title, fontsize=21)


    ax.tick_params(axis='x',labelsize=18, rotation=30, pad=10)
    ax.tick_params(axis='y',labelsize=18, rotation=60, pad=20)
    ax.tick_params(axis='z',which='both', left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False,labelsize=12)
    id1 = np.argmax(coord1[:,0])
    ax.text(coord1[id1,0]+1, coord1[id1,1]-0.5, coord1[id1,2], "PV\nNeuron", fontsize=13)
    id2 = np.argmax(coord2[:,0])
    ax.text(coord2[id2,0]+1, coord2[id2,1]-0.5, coord2[id2,2], "Pyr\nNeuron", fontsize=13)
    if target:
        ax.text(coord1[id1,0]-2.5, coord1[id1,1]-0.5, coord1[id1,2], "Target Region", fontsize=17)
    else:
        ax.text(coord1[id1, 0] - 2.5, coord1[id1, 1] - 0.5, coord1[id1, 2], "Non-Target Region", fontsize=17)
    ax.text(coord1[id1,0]-4, coord1[id1,1]+4.4, coord1[id1,2], "(Hz)", fontsize=18)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=1.5*np.min([xmin,ymin]), ymax=1.5*np.max([xmax,ymax]))
    ax.set_xlim(xmin=1.5*np.min([xmin,ymin]), xmax=1.5*np.max([xmax,ymax]))

    plt.tight_layout()
    if savepath is not None:
        ax.view_init(45,120)
        plt.savefig(savepath+'_orientation1.png')
        ax.view_init(45,240)
        plt.savefig(savepath+'_orientation2.png')
        ax.view_init(45,90)
        plt.savefig(savepath+'_orientation3.png')
        ax.view_init(45,0)
        plt.savefig(savepath+'_orientation4.png')
        ax.view_init(90,0)
        plt.savefig(savepath+'_orientation5.png')

    view_angle = np.linspace(0,360,361)
    def update(frame):
        ax.view_init(45,view_angle[frame])
    ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ani.save(os.path.join(savepath+'.gif'), writer='pillow')

    if show:
        plt.show()
    else:
        plt.close()
##################################################################################
##################################################################################
##################################################################################   

##################################################################################
################## Non-Invasive Experimental Setup ###############################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))

SAVE_PATH_ti = os.path.join(os.getcwd(),'TISimResults/TI/TI_target_region')
if not os.path.exists(SAVE_PATH_ti):
    os.makedirs(SAVE_PATH_ti)

SAVE_PATH_no_ti = os.path.join(os.getcwd(),'TISimResults/TI/TI_non-target_region')
if not os.path.exists(SAVE_PATH_no_ti):
    os.makedirs(SAVE_PATH_no_ti)


#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head

#### Defing the SAVING Directory for 2elec4cm-4cm Location
SAVE_PATH_ti = os.path.join(SAVE_PATH_ti, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH_ti):
    os.makedirs(SAVE_PATH_ti)

SAVE_PATH_no_ti = os.path.join(SAVE_PATH_no_ti, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH_no_ti):
    os.makedirs(SAVE_PATH_no_ti) 

depth = 15 ## mm

## Locations to evaluate neurons
points_samples_ti = sample_spherical(num_samples=100, theta_max=np.pi/2-1/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))
points_samples = sample_spherical(num_samples=250, theta_max=np.pi/2-7/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))
amp_level = np.load(os.path.join(SAVE_PATH_ti,"Amplitude.npy"))
for l in range(len(amp_level)):
    
    round_start_time = time.time() 

    ## Defining Saving Directories
    #########################################################################################
    SAVE_PATH_rawdata_ti = os.path.join(SAVE_PATH_ti, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH_ti, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata_ti):
        os.makedirs(SAVE_PATH_rawdata_ti)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)

    SAVE_PATH_rawdata_no_ti = os.path.join(SAVE_PATH_no_ti, 'AmpLevel'+str(l)+'/RawData')
    if not os.path.exists(SAVE_PATH_rawdata_no_ti):
        os.makedirs(SAVE_PATH_rawdata_no_ti)
    ## Loading Data
    #######################################################################################
    start_time = time.time()
    print("Loading Raw Data for Amplitude Level %d..."%(l))
    ti_region_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata_ti,'Pyr_fr.npy'))
    ti_region_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata_ti,'PV_fr.npy'))
    no_ti_region_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata_no_ti,'Pyr_fr.npy'))
    no_ti_region_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata_no_ti,'PV_fr.npy'))
    
    idx_no_ti = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))>=1
    no_ti_region_ti_pyr = no_ti_region_ti_pyr[idx_no_ti] 
    no_ti_region_ti_pv = no_ti_region_ti_pv[idx_no_ti] 
    points_samples_no_ti = points_samples[idx_no_ti]
    if int(amp_level[l]) == 1100:
        if not os.path.exists(os.path.join(cwd, 'TISimResults/Figs5Main')):
            os.makedirs(os.path.join(cwd, 'TISimResults/Figs5Main'))
        plot_response(coord=points_samples_ti, values1=ti_region_ti_pv, values2=ti_region_ti_pyr, y_displace=2, title=None, savepath=os.path.join(cwd, 'TISimResults/Figs5Main/Fig5-j'), show=True, target=True)
        plot_response(coord=points_samples_no_ti, values1=no_ti_region_ti_pv, values2=no_ti_region_ti_pyr, y_displace=6, title=None, savepath=os.path.join(cwd, 'TISimResults/Figs5Main/Fig5-k'), show=True, target=False)

    plot_response(coord=points_samples_ti, values1=ti_region_ti_pv, values2=ti_region_ti_pyr, y_displace=2, title=None, savepath=os.path.join(SAVE_PATH_plots,'TI_region_Pyr_PV_TI_Response'), show=True, target=True)
    plot_response(coord=points_samples_no_ti, values1=no_ti_region_ti_pv, values2=no_ti_region_ti_pyr, y_displace=6, title=None, savepath=os.path.join(SAVE_PATH_plots,'No_TI_region_Pyr_PV_TI_Response'), show=True, target=False)

    print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    ## Plotting Results    
    #######################################################################################
    idx_ti = ti_region_ti_pyr>5
    labels = ['PV', 'Pyr', 'PV', 'Pyr']
    idx_no_ti = no_ti_region_ti_pyr>5 
    data = [ti_region_ti_pv[idx_ti], ti_region_ti_pyr[idx_ti], no_ti_region_ti_pv[idx_no_ti], no_ti_region_ti_pyr[idx_no_ti]]
    
    if np.sum(idx_ti)>0 and np.sum(idx_no_ti)>0:
        x = []
        x_pos = np.array([0,1,3,4])
        for i in range(len(labels)):
            x.append(np.random.normal(x_pos[i], 0.04, data[i].shape[0]))
        clevel = np.linspace(0,1,len(data))
        plt.boxplot(data, labels=labels, positions=x_pos)
        for i in range(len(data)):
            plt.scatter(x[i], data[i].flatten(), c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
        plt.ylabel("Firing Rate (Hz)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.ylim(ymin=0)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,"TI_comparison.png"))
        plt.show()


