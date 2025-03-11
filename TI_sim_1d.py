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
import matplotlib as mpl
from helper import cart_to_sph, sph_to_cart, sample_1d
mpl.rcParams['figure.dpi'] = 600

### Helper Functions
##################################################################################
##################################################################################
##################################################################################


def plot_points_to_sample(coord_elec1, coord_elec2, J1, J2, freq1, freq2, points, savepath):
    
    skull_samples = np.random.normal(loc=0, scale=1, size=(10**4,3))
    skull_samples = skull_samples/np.sqrt(np.sum(skull_samples**2, axis=1)).reshape(-1,1)*(9.2-0.6)
    skull_samples = cart_to_sph(skull_samples)
    skull_samples = skull_samples[skull_samples[:,1]>(np.pi/2-7/9.2)]
    skull_samples = sph_to_cart(skull_samples)
    scalp_samples = skull_samples.copy()/(9.2-0.6)*(9.2)
    csf_samples = skull_samples.copy()/(9.2-0.6)*(9.2-1.1)
    brain_samples = skull_samples.copy()/(9.2-0.6)*(9.2-1.2)
    
    skull_samples =  skull_samples[skull_samples[:,1]<=2]
    skull_samples =  skull_samples[skull_samples[:,1]>=-2]

    scalp_samples =  scalp_samples[scalp_samples[:,1]<=2]
    scalp_samples =  scalp_samples[scalp_samples[:,1]>=-2]
    
    csf_samples =  csf_samples[csf_samples[:,1]<=2]
    csf_samples =  csf_samples[csf_samples[:,1]>=-2]

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    img = ax.scatter(coord_elec1[:,0], coord_elec1[:,1], coord_elec1[:,2], linewidth=0.3, s=100, color='darkred', label=str(round(freq1))+" Hz")
    img = ax.scatter(coord_elec2[:,0], coord_elec2[:,1], coord_elec2[:,2], linewidth=0.3, s=100, color='darkgreen', label=str(round(freq2))+" Hz")
    img = ax.scatter(skull_samples[:,0], skull_samples[:,1], skull_samples[:,2], linewidth=0.3, s=10, color='grey', alpha=0.1)
    img = ax.scatter(scalp_samples[:,0], scalp_samples[:,1], scalp_samples[:,2], linewidth=0.3, s=10, color='salmon', alpha=0.1)
    img = ax.scatter(csf_samples[:,0], csf_samples[:,1], csf_samples[:,2], linewidth=0.3, s=10, color='deepskyblue', alpha=0.1)
    img = ax.scatter(brain_samples[:,0], brain_samples[:,1], brain_samples[:,2], linewidth=0.3, s=10, color='crimson',alpha=0.1)
    target_points = points[np.sum(points[:,:2]**2, axis=1)<=1]
    off_target_points = points[np.sum(points[:,:2]**2, axis=1)>=1]
    img = ax.scatter(target_points[:,0], target_points[:,1], target_points[:,2], linewidth=0.3, s=30, color='orange')
    img = ax.scatter(off_target_points[:,0], off_target_points[:,1], off_target_points[:,2], linewidth=0.3, s=30, color='blue')

    ax.set_xlabel('X-axis (cm)', fontsize=14)
    ax.set_ylabel('Y-axis (cm)', fontsize=14)
    ax.set_zlabel('Z-axis (cm)', fontsize=14)
    ax.set_title('Locations of Points Evaluated', fontsize=21)
    for i in range(coord_elec1.shape[0]):
        if J1[i]>0:
            ax.text(coord_elec1[i,0]+1.2,coord_elec1[i,1],coord_elec1[i,2]+0.2, str(round(J1[i],3)), fontsize=11)
            ax.text(coord_elec1[i,0]+2.7,coord_elec1[i,1],coord_elec1[i,2]+0.7, '2000 Hz', fontsize=11)
        else:
            ax.text(coord_elec1[i,0]+1.2,coord_elec1[i,1],coord_elec1[i,2]+0.2,str(round(J1[i],3)), fontsize=11)
    for i in range(coord_elec1.shape[0]):
        if J2[i]>0:
            ax.text(coord_elec2[i,0],coord_elec2[i,1],coord_elec2[i,2]+0.2, str(round(J2[i],3)), fontsize=11)
            ax.text(coord_elec2[i,0]-0.3,coord_elec2[i,1],coord_elec2[i,2]+0.7, '2020 Hz', fontsize=11)
        else:
            ax.text(coord_elec2[i,0],coord_elec2[i,1],coord_elec2[i,2]+0.2,str(round(J2[i],3)), fontsize=11)

    ax.text(1.2,0,9.2-0.3, 'Scalp', fontsize=11)
    ax.text(1.2,0,9.2-0.6-0.3, 'Skull', fontsize=11)
    ax.text(-2.8,-4,7.1, 'CSF', fontsize=11)
    ax.text(1.2,0,9.2-2.8, 'Brain', fontsize=11)

    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z',labelsize=12)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=np.min([xmin,ymin]), ymax=np.max([xmax,ymax]))
    ax.set_xlim(xmin=np.min([xmin,ymin]), xmax=np.max([xmax,ymax]))


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
        ax.view_init(25,view_angle[frame])
    ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
    plt.show()
    plt.close()

##################################################################################
##################################################################################
##################################################################################   

##################################################################################
######################## TI Experimental Setup ###################################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/TI/1-D')
#SAVE_PATH = "Garbage"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head
freq1, freq2 = int(sys.argv[1]), int(sys.argv[2]) ## Deciding the two frequencies for TI

#### Defing the SAVING Directory for 2elec4cm-4cm Location
SAVE_PATH = os.path.join(SAVE_PATH, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining 2000Hz Electrode Configuration
#################################################################################
theta_patch_2000 = np.pi/2-np.array([6/overall_radius,2/overall_radius]).reshape(-1,1)
phi_patch_2000 =  np.array([0,0]).reshape(-1,1)
cart_patch_2000 = sph_to_cart(np.hstack([overall_radius*np.ones([len(theta_patch_2000),1]), theta_patch_2000, phi_patch_2000]))
print("Cartesian Coordinates of Anode ([x,y,z]) for the 2000 Hz electrodes: %s cm"%(str(np.round(cart_patch_2000[1],2))))
print("Cartesian Coordinates of Cathode ([x,y,z]) for the 2000 Hz electrodes: %s cm"%(str(np.round(cart_patch_2000[0],2))))
## Describing the relative portions of current going through each electrode, the postive and negative components should sum upto 1 and -1 respectively
J_2000 = np.array([-1,1])*0.5

## Saving Electric Field Values, so we can reload them
elec_dir = os.path.join(os.getcwd(),'extracellular_voltage')
if not os.path.exists(elec_dir):
    os.makedirs(elec_dir)

file_dir_2000 = os.path.join(elec_dir,'human_elec_2elec4cm-4cm_2000kHz')
if os.path.exists(file_dir_2000+"_r.npy"):
    fname_load_2000 = file_dir_2000
    fname_save_2000 = None
else:
    fname_save_2000 = file_dir_2000
    fname_load_2000 = None
depth = 15 ## mm
elec_field_2000 = ray.put(sparse_place_human(r=depth-1.5, J=J_2000, fname_save=fname_save_2000, fname_load=fname_load_2000, theta_patch=theta_patch_2000, phi_patch=phi_patch_2000))

#### Defining 2020Hz Electrode Configuration
#################################################################################
theta_patch_2020 = np.pi/2-np.array([6/overall_radius,2/overall_radius]).reshape(-1,1)
phi_patch_2020 =  np.array([np.pi,np.pi]).reshape(-1,1)
cart_patch_2020 = sph_to_cart(np.hstack([overall_radius*np.ones([len(theta_patch_2020),1]), theta_patch_2020, phi_patch_2020]))
print("Cartesian Coordinates of Anode ([x,y,z]) for the 2020 Hz electrodes: %s cm"%(str(np.round(cart_patch_2020[1],2))))
print("Cartesian Coordinates of Cathode ([x,y,z]) for the 2020 Hz electrodes: %s cm"%(str(np.round(cart_patch_2020[0],2))))
## Describing the relative portions of current going through each electrode, the postive and negative components should sum upto 1 and -1 respectively
J_2020 = np.array([-1,1])*0.5

## Saving Electric Field Values, so we can reload them 
file_dir_2020 = os.path.join(elec_dir,'human_elec_2elec4cm-4cm_2020kHz')
if os.path.exists(file_dir_2020+"_r.npy"):
    fname_load_2020 = file_dir_2020
    fname_save_2020 = None
else:
    fname_save_2020 = file_dir_2020
    fname_load_2020 = None

elec_field_2020 = ray.put(sparse_place_human(r=depth-1.5, J=J_2020, fname_save=fname_save_2020, fname_load=fname_load_2020, theta_patch=theta_patch_2020, phi_patch=phi_patch_2020))
## Locations to evaluate neurons
points_samples = sample_1d(num_samples=50, theta_max=7/overall_radius, r=overall_radius-depth*10**(-1))


savepath = os.path.join(SAVE_PATH,"ExpSetup")
if not os.path.exists(savepath):
    os.makedirs(savepath)
plot_points_to_sample(coord_elec1=cart_patch_2000, coord_elec2=cart_patch_2020, J1=J_2000, J2=J_2020, freq1=freq1, freq2=freq2, points=points_samples, savepath=os.path.join(savepath, "SampledPoints"))

print("Electric Field Simulator Loaded! Time Taken %s s"%(str(round(time.time()-start_time,3))))

#### Defining Variables for Setting up Simulation
################################################################################

cell_id_pyr_lst =np.array([6]) #np.array([6,7,8,9,10]) ## Different Morphology for L23 Pyr Cells
cell_id_pv_lst = np.array([33])#np.array([32,33,34,35,36]) ## Different Morphology for L23 LBC Cells
human_or_mice = ray.put(1) ## 1->mice, 0-> human
temp = ray.put(34.0) ## Celsius, temparature at which neurons are simulated
dt = ray.put(0.025) ## ms, discretization time step
num_cores = 50 ## Number of Cores used for Parallelization
SHOW_PLOTS = False ## Flag used for showing or not showing plots

#### Non-Invasive Stimulation
###################################################################################
###################################################################################

## Generating Waveforms
start_time, time_taken_round = time.time(), 0
print("Generating Waveform...")
pulse_train_sin = PulseTrain_Sinusoid()
total_time, sampling_rate = 2000, 1e6 ## ms, Hz
amp_array_2000, time_array = pulse_train_sin.amp_train(amp=1, freq=freq1, total_time=total_time, sampling_rate=sampling_rate)
amp_array_2020, _ = pulse_train_sin.amp_train(amp=1, freq=freq2, total_time=total_time, sampling_rate=sampling_rate)
Efield_2000 = ray.get(elec_field_2000)._calc_Efield(x=points_samples[:,0].reshape(-1,1)*10,y=points_samples[:,1].reshape(-1,1)*10,z=points_samples[:,2].reshape(-1,1)*10)
Efield_2020 = ray.get(elec_field_2020)._calc_Efield(x=points_samples[:,0].reshape(-1,1)*10,y=points_samples[:,1].reshape(-1,1)*10,z=points_samples[:,2].reshape(-1,1)*10)
Efield_norm = np.empty(len(points_samples))
Efield_r = np.empty(len(points_samples))
Efield_theta = np.empty(len(points_samples))
Efield_phi = np.empty(len(points_samples))

for i in range(len(points_samples)):
    Efield_r[i] = np.max(amp_array_2000*Efield_2000[i,0]+amp_array_2020*Efield_2020[i,0])
    Efield_theta[i] = np.max(amp_array_2000*Efield_2000[i,1]+amp_array_2020*Efield_2020[i,1])
    Efield_phi[i] = np.max(amp_array_2000*Efield_2000[i,2]+amp_array_2020*Efield_2020[i,2])
Efield_norm = np.sqrt(Efield_r**2+Efield_theta**2+Efield_phi**2)
plt.plot(points_samples[:,0], Efield_r, marker='x', label='r-dir')
plt.plot(points_samples[:,0], Efield_theta, marker='x', label='theta-dir')
plt.plot(points_samples[:,0], Efield_phi, marker='x', label='phi-dir')
plt.plot(points_samples[:,0], Efield_norm, marker='x', label='Norm')
plt.xlabel('Amplitude (mA/cm^2)', fontsize=19)
plt.ylabel('X-axis (cm)', fontsize=19)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "ElectricField.png"))
plt.show()

amp_array_2000, amp_array_2020 =ray.put(amp_array_2000), ray.put(amp_array_2020)
time_array = ray.put(time_array)
sampling_rate = ray.put(sampling_rate)
save_state_show = ray.put(False)
print("Waveform Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))
LOAD_DATA_FLAG = False
if not LOAD_DATA_FLAG:
    min_level, max_level = float(sys.argv[3]), float(sys.argv[4])
    amp_level = np.linspace(min_level, max_level, 4)
    np.save(os.path.join(SAVE_PATH,"Amplitude.npy"), amp_level)
    sim_already_performed = 0 
else:
    amp_level = np.load(os.path.join(SAVE_PATH,'Amplitude.npy'))
    sim_already_performed = 0
    
for l in range(sim_already_performed,len(amp_level)):
    
    round_start_time = time.time()    
    #### Defining Saving Directories
    #########################################################################################
    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)


    if not LOAD_DATA_FLAG:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        print("Starting Simulation for Amplitude Level %d"%(l))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
     
        split_num = int(np.floor(points_samples.shape[0]/num_cores)) ## Defines How Many Simulations Can be run in parallel

        #### Defining Locations and Orientation of Pyr and PV neurons to be evaluated
        angle = cart_to_sph(points_samples)
        angle[:,1] = np.pi/2-angle[:,1]
        angle_pyr = np.hstack([angle[:,1].copy().reshape(-1,1),angle[:,2].copy().reshape(-1,1)]) ## parameter used for specifying rotation of Pyr morphology
        angle_pv = np.hstack([angle[:,1].copy().reshape(-1,1),angle[:,2].copy().reshape(-1,1)]) ## parameter used for specifying rotation of PV morphology
        angle_pyr = np.array_split(angle_pyr, split_num, axis=0)
        angle_pv = np.array_split(angle_pv, split_num, axis=0)
        
        loc_pyr = np.array_split(points_samples*10**4, split_num, axis=0) ## cm->um, parameter used for specifying location of Pyr morphology
        loc_pv = np.array_split(points_samples*10**4, split_num, axis=0) ## cm->um, parameter used for specifying location of PV morphology
        
        ### Run Pyr Stimulation
        ######################################################################################
        cell_id_pyr = cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst), size=points_samples.shape[0])] ## Randomly choosing a Pyr Morphology out of the 5 available
        cell_id_pyr = np.array_split(cell_id_pyr, split_num, axis=0)
        start_time = time.time()
        print("Simulation for Pyr Neuron Started...")
        fr_rate_ti_pyr = [] 
        for num in range(split_num):
            neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr[num][i], temp=temp, dt=dt, elec_field=elec_field_2000, elec_field2=elec_field_2020) for i in range(loc_pyr[num].shape[0])] ## Initializing neuron model
            ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pyr[num][i], pos_neuron=loc_pyr[num][i]) for i in range(len(neuron))]) ## Setting Extracellular Stim Paramaters
            delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation
            
            ## TI Stimulation
            results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_2000, amp_array2=amp_array_2020, scale1=amp_level[l], scale2=amp_level[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]
            results = ray.get(results)
            fr_rate_ti_pyr.append(np.array([results[i][1] for i in range(num_cores)]).flatten())
            ## Uncomment to see plots of membrane potential
            #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
            del neuron
        
        fr_rate_ti_pyr = np.hstack(fr_rate_ti_pyr)
        fr_rate_ti_pyr = fr_rate_ti_pyr/(total_time*1e-03)
        print("Pyr Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

 
        #### Run PV Stimulation
        ######################################################################################
        cell_id_pv = cell_id_pv_lst[np.random.randint(len(cell_id_pv_lst), size=points_samples.shape[0])] ## Randomly choosing a Pyr Morphology out of the 5 available
        cell_id_pv = np.array_split(cell_id_pv, split_num, axis=0)
        
        start_time = time.time()
        print("Simulation for PV Neuron Started...")
        fr_rate_ti_pv = []
        for num in range(split_num):
            neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv[num][i], temp=temp, dt=dt, elec_field=elec_field_2000, elec_field2=elec_field_2020) for i in range(loc_pyr[num].shape[0])] ## Initializing neuron model
            ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pyr[num][i], pos_neuron=loc_pyr[num][i]) for i in range(len(neuron))]) ## Setting Extracellular Stim Paramaters
            delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation
            
            ## TI Stimulation
            results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_2000, amp_array2=amp_array_2020, scale1=amp_level[l], scale2=amp_level[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]
            results = ray.get(results)
            fr_rate_ti_pv.append(np.array([results[i][1] for i in range(num_cores)]).flatten())
            ## Uncomment to see plots of membrane potential
            #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
            del neuron

        fr_rate_ti_pv = np.hstack(fr_rate_ti_pv)
        fr_rate_ti_pv = fr_rate_ti_pv/(total_time*1e-03)
        print("PV Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

        ## Saving Data
        #######################################################################################
        start_time = time.time()
        print("Saving Raw Data for Amplitude Level %d..."%(l))
        np.save(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'), fr_rate_ti_pyr)
        np.save(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'), fr_rate_ti_pv)     
        print("Raw Data Saved for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    
    else:
        
        ## Loading Data
        #######################################################################################
        start_time = time.time()
        print("Loading Raw Data for Amplitude Level %d..."%(l))
        fr_rate_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'))
        fr_rate_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'))
        print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

    ## Plotting Results    
    #######################################################################################
    
    time_taken_round = time_taken_round*(l)/(l+1)+(time.time()-round_start_time)/(l+1)
    ETA = ((len(amp_level)-l-1)*time_taken_round)/3600
    print("Simulation Finished for Amplitude Level %d! Time Taken %s hr. ETA for script to finish: %s hr"%(l, str(round((time.time()-round_start_time)/3600,3)),str(round(ETA,3))))
    idx_ti = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))<1 
    idx_no_ti = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))>=1  
    plt.plot(points_samples[:,0], fr_rate_ti_pyr, marker='x', label='Pyr',color='blue')
    plt.plot(points_samples[:,0], fr_rate_ti_pv, marker='x', label='PV', color='orange')
    plt.vlines(1, ymin=0, ymax=np.max([np.max(fr_rate_ti_pv), np.max(fr_rate_ti_pyr)]), linestyle='--', color='black')
    plt.vlines(-1, ymin=0, ymax=np.max([np.max(fr_rate_ti_pv), np.max(fr_rate_ti_pyr)]), linestyle='--', color='black')
    plt.legend(fontsize=17, loc='upper center')
    plt.xlabel('X-axis (cm)', fontsize=19)
    plt.ylabel('Firing Rate (Hz)', fontsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.tight_layout()
    if int(amp_level[l])==1100:
        if not os.path.exists(os.path.join(cwd, 'TISimResults/Figs5Main')):
            os.makedirs(os.path.join(cwd, 'TISimResults/Figs5Main'))
        plt.savefig(os.path.join(cwd, 'TISimResults/Figs5Main/Fig5-l.png'))
    plt.savefig(os.path.join(SAVE_PATH_plots,'1d-TI_PV_Pyr_FR.png'))
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    
