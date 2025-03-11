import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
from elec_field import UniformField
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI
import sys
import math

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
    for i in range(samples):
        y = 1-(i/float(samples-1))*2  # y goes from 1 to -1
        radius = math.sqrt(1-y*y)  # radius at y
        theta = phi*i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)

##################################################################################
##################################################################################
##################################################################################   

##################################################################################
################## Uniform Experimental Setup ####################################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
#### Defining Variables for Setting up Simulation

cell_id_pyr_lst = [6,7,8,9,10] ## Different Morphology for L23 Pyr Cells
cell_id_pv_lst = [32,33,34,35,36] ## Different Morphology for L23 LBC Cells
human_or_mice = ray.put(1) ## 1->mice, 0-> human
temp = ray.put(34.0) ## Celsius, temparature at which neurons are simulated
dt = ray.put(0.025) ## ms, discretization time step
num_cores = 30 ## Number of Cores used for Parallelization
SHOW_PLOTS = False ## Flag used for showing or not showing plots
unit_vec = fibonacci_sphere(samples=30) ## Sampling 20 approximately uniformly spaced unit direction vectors along the electrode locations from ICMS study

angle_pv = np.array([0,0]) ## parameter used for specifying rotation of PV morphology
angle_pyr = np.array([0,0]) ## parameter used for specifying rotation of Pyr morphology

loc_pyr = np.array([0,0,0]) ## parameter used for specifying location of Pyr morphology
loc_pv = np.array([0,0,0]) ## parameter used for specifying location of PV morphology

#### Plotting Directions and Neurons
###################################################################################
###################################################################################
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/PointElectrodeSim/Results_uniform')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

PLOT_NEURON_AND_UNITVEC = True
if PLOT_NEURON_AND_UNITVEC:
    def plot_electrode_and_neuron(coord_elec, coord, savepath=None):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        ax.quiver(coord_elec[:,0]*1.2, coord_elec[:,1]*1.2, coord_elec[:,2]*1.2, coord_elec[:,0], coord_elec[:,1], coord_elec[:,2], length=0.2, linewidth=1.0, color='orange')
        ax.scatter(coord[:,0]*10**(-3),coord[:,1]*10**(-3),coord[:,2]*10**(-3), linewidth=1.0, s=2.0)
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
        ax.set_title('Neuron Orientation w.r.t\n Uniform Field ', fontsize=21)

        view_angle = np.linspace(0,360,361)
        def update(frame):
            ax.view_init(10,view_angle[frame])
        ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
        ani.save(os.path.join(savepath+'.gif'), writer='pillow')
        ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
        plt.show()

    ################### Plot Pyr Coordinates ##################################################################

    for cell_id_pyr in cell_id_pyr_lst:
        
        ## Get Neuron Coordinates
        neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt)
        coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pyr, angle=angle_pyr))
        
        ## Plot ICMS Electrode With Neuron 
        savepath_curr = os.path.join(SAVE_PATH,'NeuronOrientation_Uniform_cellid'+str(cell_id_pyr)+'.png')
        plot_electrode_and_neuron(coord_elec=unit_vec, coord=coord, savepath=savepath_curr)
        
        del neuron   
    
    ################### Plot PV Coordinates ##################################################################
    
    for cell_id_pv in cell_id_pv_lst:
        ## Get Neuron Coordinates
        neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv, temp=temp, dt=dt)
        coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pyr, angle=angle_pv))
        
        ## Plot ICMS Electrode With Neuron 
        savepath_curr = os.path.join(SAVE_PATH,'NeuronOrientation_Uniform_cellid'+str(cell_id_pv)+'.png')
        plot_electrode_and_neuron(coord_elec=unit_vec, coord=coord, savepath=savepath_curr)
        
        del neuron
    
    ############################################################################################################

#### Uniform Stimulation
###################################################################################
###################################################################################

## Generating Waveforms
start_time, time_taken_round = time.time(), 0
print("Generating Waveform...")
pulse_train_sin, pulse_train_ti = PulseTrain_Sinusoid(), PulseTrain_TI()
freq1, freq2 = 2*1e3, 2.02*1e3 ## Hz
total_time, sampling_rate = 2000, 1e6 ## ms, Hz
amp_array_sin, time_array = pulse_train_sin.amp_train(amp=1, freq=freq1, total_time=total_time, sampling_rate=sampling_rate)
amp_array_ti, _ = pulse_train_ti.amp_train(amp1=0.5, amp2=0.5, freq1=freq1, freq2=freq2, total_time=total_time, sampling_rate=sampling_rate)
amp_arra_sin, amp_array_ti, amp_array2 =ray.put(amp_array_sin), ray.put(amp_array_ti),ray.put(np.zeros(time_array.shape))
time_array = ray.put(time_array)
sampling_rate = ray.put(sampling_rate)
save_state_show = ray.put(False)
print("Waveform Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))
activ_thresh_pyr, activ_thresh_pv = np.empty(len(unit_vec)), np.empty(len(unit_vec))
fr_activ_pyr, fr_activ_pv = np.empty([len(unit_vec),4]), np.empty([len(unit_vec), 4])
LOAD_DATA_FLAG = False
cell_id_rand_pyr, cell_id_rand_pv = [], [] 
for l in range(len(unit_vec)):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
    print("Starting Simulation for Electrode Location %d"%(l))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
    round_start_time = time.time() 
    
    ## Defining Directories for Saving DATA
    #########################################################################################
    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'elecloc'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'elecloc'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)
    
    if not LOAD_DATA_FLAG:
        ## Generate Electric Field Simulator
        start_time = time.time()
        print("Loading Electric Field Simulator...")
        elec_field, elec_field2 = ray.put(UniformField(unit_vec=unit_vec[l])), ray.put(None)
        print("Electric Field Simulator Loaded! Time Taken %s s"%(str(round(time.time()-start_time,3))))
        
        ### Run Pyr Stimulation
        ######################################################################################
        cell_id_pyr = ray.put(cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst))]) ## Randomly choosing a Pyr Morphology out of the 5 available
        neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt, elec_field=elec_field, elec_field2=elec_field2) for i in range(2*num_cores)] ## Initializing neuron model
        print("Pyramidal Cell Id chosen %d."%(int(ray.get(cell_id_pyr))))
        ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pyr, pos_neuron=loc_pyr) for i in range(2*num_cores)]) ## Setting Extracellular Stim Paramaters
        delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation

        ## Checking if Save State for Pyramidal neuron model Exists and if not then creating one
        save_state = os.path.join(cwd,'cells/SaveState/human_or_mice'+str(ray.get(human_or_mice))+'cell-'+str(ray.get(cell_id_pyr))+'_Temp-'+str(ray.get(temp))+'C_dt-'+str(ray.get(dt)*10**3)+'us_delay-'+str(ray.get(delay_init))+'ms.bin')
        if not os.path.exists(save_state):
            start_time = time.time()
            print("Generating Save State for Pyr Neuron...")
            ray.get(neuron[0].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, amp_array2=amp_array2, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show))
            print("Save State Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))
   
        ## Deciding the range of amplitude across which to stimulate neurons
        min_amp_pyr, max_amp_pyr = float(sys.argv[1]), float(sys.argv[2]) ## uA
        while True:
            amp_rough = np.linspace(min_amp_pyr, max_amp_pyr, 30)
            results_rough = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=amp_rough[i], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show) for i in range(30)]
            results_rough = ray.get(results_rough)
            fr_rate = np.array([results_rough[i][1] for i in range(30)]).flatten()
            fr_rate = fr_rate/(total_time*1e-03)
            amp_rough = amp_rough[fr_rate-20>0]
            fr_rate = fr_rate[fr_rate-20>0] 
            if fr_rate.size == 0:
                max_amp_pyr = max_amp_pyr*1.1
                print("Maximum Threshold Too Low!!! Increasing it by 10%")
            else:
                break
        max_amp_pyr = amp_rough[np.argmin(fr_rate)]*1.05
        print("Minimum and Maximum Threshold adjusted to %s V/m and %s V/m"%(str(round(min_amp_pyr,3)),str(round(max_amp_pyr,3))))
        mono_ICMS_amp_pyr = np.linspace(min_amp_pyr, max_amp_pyr, num_cores) ## uA 
        np.save(os.path.join(SAVE_PATH_rawdata,'Amplitude_Pyr.npy'), mono_ICMS_amp_pyr)
        
        ## Providing pure and modulated sinusoidal stimulation
        ########################################################################################
        start_time = time.time()
        print("Simulation for Pyr Neuron Started...")
        results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, scale1=mono_ICMS_amp_pyr[i], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]+[neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=mono_ICMS_amp_pyr[i-num_cores], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show) for i in range(num_cores,2*num_cores)]
        results = ray.get(results)
        fr_rate_sin_pyr = np.array([results[i][1] for i in range(num_cores)]).flatten()
        fr_rate_sin_pyr = fr_rate_sin_pyr/(total_time*1e-03)
        fr_rate_ti_pyr = np.array([results[i][1] for i in range(num_cores,2*num_cores)]).flatten()
        fr_rate_ti_pyr = fr_rate_ti_pyr/(total_time*1e-03)
        print("Pyr Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

        ## Uncomment to see plots of membrane potential
        #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
        #########################################################################################

        del neuron
        
        #### Run PV Stimulation
        ######################################################################################
        cell_id_pv = ray.put(cell_id_pv_lst[np.random.randint(len(cell_id_pv_lst))]) ## Randomly choosing a PV Morphology out of the 5 available
        neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv, temp=temp, dt=dt, elec_field=elec_field, elec_field2=elec_field2) for i in range(2*num_cores)] ## Initializing neuron model
        print("PV Cell Id chosen %d."%(int(ray.get(cell_id_pv))))
        ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pv, pos_neuron=loc_pv) for i in range(2*num_cores)]) ## Setting Extracellular Stim Paramaters
        delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation

        ## Checking if Save State for Pyramidal neuron model Exists and if not then creating one
        save_state = os.path.join(cwd,'cells/SaveState/human_or_mice'+str(ray.get(human_or_mice))+'cell-'+str(ray.get(cell_id_pv))+'_Temp-'+str(ray.get(temp))+'C_dt-'+str(ray.get(dt)*10**3)+'us_delay-'+str(ray.get(delay_init))+'ms.bin')
        if not os.path.exists(save_state):
            start_time = time.time()
            print("Generating Save State for PV Neuron...")
            ray.get(neuron[0].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, amp_array2=amp_array2, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show))
            print("Save State Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))   
    
        ## Deciding the range of amplitude across which to stimulate neurons
        min_amp_pv, max_amp_pv = float(sys.argv[3]), float(sys.argv[4]) ## uA
        while True:
            amp_rough = np.linspace(min_amp_pv, max_amp_pv, 30)
            results_rough = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=amp_rough[i], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show) for i in range(30)]
            results_rough = ray.get(results_rough)
            fr_rate = np.array([results_rough[i][1] for i in range(30)]).flatten()
            fr_rate = fr_rate/(total_time*1e-03)
            amp_rough = amp_rough[fr_rate-20>0]
            fr_rate = fr_rate[fr_rate-20>0] 
            if fr_rate.size == 0:
                max_amp_pv = max_amp_pv*1.1
                print("Maximum Threshold Too Low!!! Increasing it by 10%")
            else:
                break
        max_amp_pv = amp_rough[np.argmin(fr_rate)]*1.05
        print("Minimum and Maximum Threshold adjusted to %s V/m and %s V/m"%(str(round(min_amp_pv,3)),str(round(max_amp_pv,3))))
        mono_ICMS_amp_pv = np.linspace(min_amp_pv, max_amp_pv, num_cores) ## uA 
        np.save(os.path.join(SAVE_PATH_rawdata,'Amplitude_PV.npy'), mono_ICMS_amp_pv)   
        
        ## Providing pure and modulated sinusoidal stimulation
        ########################################################################################
        start_time = time.time()
        print("Simulation for PV Neuron Started...")
        results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, scale1=mono_ICMS_amp_pv[i], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show) for i in range(num_cores)]+[neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=mono_ICMS_amp_pv[i-num_cores], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores, 2*num_cores)]
        results = ray.get(results)
        fr_rate_sin_pv = np.array([results[i][1] for i in range(num_cores)]).flatten()
        fr_rate_sin_pv = fr_rate_sin_pv/(total_time*1e-03)
        fr_rate_ti_pv = np.array([results[i][1] for i in range(num_cores, 2*num_cores)]).flatten()
        fr_rate_ti_pv = fr_rate_ti_pv/(total_time*1e-03)
        print("PV Simulation Performed! Time Taken %s s"%(str(round(time.time()-start_time,3))))

        ## Uncomment to see plots of membrane potential
        #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
        #########################################################################################

        del neuron

        ## Saving Data
        #######################################################################################
        start_time = time.time()
        print("Saving Raw Data for Direction %d..."%(l))
        np.save(os.path.join(SAVE_PATH_rawdata,'Pyr_sin_fr.npy'), fr_rate_sin_pyr)
        np.save(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'), fr_rate_ti_pyr)
        np.save(os.path.join(SAVE_PATH_rawdata,'PV_sin_fr.npy'), fr_rate_sin_pv)
        np.save(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'), fr_rate_ti_pv)
        print("Raw Data Saved for Direction %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

        ## Plotting Results    
        #######################################################################################
        
        start_time = time.time()
        print("Plotting Data for Direction %d..."%(l))
        
        ## Cell-Type Comparison for Cell-Type Comparison 
        plt.plot(mono_ICMS_amp_pyr*1e-03, fr_rate_sin_pyr, marker='x', label='Pyr', color='red')
        plt.plot(mono_ICMS_amp_pv*1e-03, fr_rate_sin_pv, marker='x', label='PV', color='green')
        plt.title('Pure Sinusoidal Stimulation:\nCell Type Comparison', fontsize=22)
        plt.legend(fontsize=15)
        plt.xlabel('Amp of Injected Current (mA)', fontsize=19)
        plt.ylabel('Firing Rate (Hz)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,'Cell_Type_Sinusoid.png'))
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        ## Cell-Type Comparison for TI-Type Comparison 
        plt.plot(mono_ICMS_amp_pyr*1e-03, fr_rate_ti_pyr, marker='x', label='Pyr', color='red')
        plt.plot(mono_ICMS_amp_pv*1e-03, fr_rate_ti_pv, marker='x', label='PV', color='green')
        plt.title('Modulated Sinusoidal Stimulation:\nCell Type Comparison', fontsize=22)
        plt.legend(fontsize=15)
        plt.xlabel('Amp of Injected Current (mA)', fontsize=19)
        plt.ylabel('Firing Rate (Hz)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,'Cell_Type_TI.png'))
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        ## Sinusoid-TI Comparison for Pyr Neuron
        plt.plot(mono_ICMS_amp_pyr*1e-03, fr_rate_sin_pyr, marker='x', label='Pure', color='blue')
        plt.plot(mono_ICMS_amp_pyr*1e-03, fr_rate_ti_pyr, marker='x', label='Modulated', color='orange')
        plt.title('Pure V/s Modulated Sinusoidal\n Stimulation: Pyramidal Cell Type', fontsize=22)
        plt.legend(fontsize=15)
        plt.xlabel('Amp of Injected Current (mA)', fontsize=19)
        plt.ylabel('Firing Rate (Hz)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,'Pyr_Sin_TI.png'))
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        ## Sinusoid-TI Comparison for PV Neuron
        plt.plot(mono_ICMS_amp_pv*1e-03, fr_rate_sin_pv, marker='x', label='Pure', color='blue')
        plt.plot(mono_ICMS_amp_pv*1e-03, fr_rate_ti_pv, marker='x', label='Modulated', color='orange')
        plt.title('Pure V/s Modulated Sinusoidal\n Stimulation: PV Cell Type', fontsize=22)
        plt.legend(fontsize=15)
        plt.xlabel('Amp of Injected Current (mA)', fontsize=19)
        plt.ylabel('Firing Rate (Hz)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,'PV_Sin_TI.png'))
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        cell_id_rand_pyr.append(cell_id_pyr)
        cell_id_rand_pv.append(cell_id_pv)
        print("Plots Saved for Direction %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    
    else:
        ## Loading Data
        #########################################################################################
        start_time = time.time()
        print("Loading Raw Data for Electrode location %d..."%(l))
        mono_ICMS_amp_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Amplitude_Pyr.npy'))
        mono_ICMS_amp_pv = np.load(os.path.join(SAVE_PATH_rawdata,'Amplitude_PV.npy'))
        fr_rate_sin_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_sin_fr.npy'))
        fr_rate_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'))
        fr_rate_sin_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_sin_fr.npy'))
        fr_rate_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'))
        cell_id_rand_pyr.append(cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst))])
        cell_id_rand_pv.append(cell_id_pv_lst[np.random.randint(len(cell_id_pyr_lst))])
        print("Raw Data Loaded for Electrode location %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

    ## Calculating activation thresholds
    #########################################################################################
    ## Activation threshold is defined as the minimum amplitude at which the neuron fires at 
    ## the beat frequency (20 Hz) of the modulated sinusoid based on the experimental results
    
    start_time = time.time()
    print("Calculation for Activation Threshold Started...")
    fr_thresh_pyr, fr_thresh_pv = fr_rate_ti_pyr[np.argmin(np.abs(fr_rate_ti_pyr-20))], fr_rate_ti_pv[np.argmin(np.abs(fr_rate_ti_pv-20))] 
    print("Actual Firing Rate Thresholds Calculated for Direction %d: %s Hz (Pyr) %s Hz (PV)"%(l,str(round(fr_thresh_pyr,3)),str(round(fr_thresh_pv,3))))
    idx_activ_pyr, idx_activ_pv = np.min(mono_ICMS_amp_pyr[np.abs(fr_rate_ti_pyr-fr_thresh_pyr)<0.5]), np.min(mono_ICMS_amp_pv[np.abs(fr_rate_ti_pv-fr_thresh_pv)<0.5])
    idx_activ_pyr, idx_activ_pv = np.argmin(np.abs(mono_ICMS_amp_pyr-idx_activ_pyr)), np.argmin(np.abs(mono_ICMS_amp_pv-idx_activ_pv))
    activ_thresh_pyr[l], activ_thresh_pv[l] = mono_ICMS_amp_pyr[idx_activ_pyr], mono_ICMS_amp_pv[idx_activ_pv]
    
    if not LOAD_DATA_FLAG:
        ## Calculating the firing rate of the counterpart neuron-type at the activation threshold of the other neuron-type
        elec_field, elec_field2 = ray.put(UniformField(unit_vec=unit_vec[l])), ray.put(None)
        neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_rand_pv[l], temp=temp, dt=dt, elec_field=elec_field, elec_field2=elec_field2) for i in range(2)]
        neuron = neuron+[NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_rand_pyr[l], temp=temp, dt=dt, elec_field=elec_field, elec_field2=elec_field2) for i in range(2)]
        ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pv, pos_neuron=loc_pv) for i in range(2)]) ## Setting Extracellular Stim Paramaters
        ray.get([neuron[i+2]._set_xtra_param.remote(angle=angle_pyr, pos_neuron=loc_pyr) for i in range(2)])
        delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation


        results = [neuron[0].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, scale1=activ_thresh_pyr[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final)]
        results =results+[neuron[1].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=activ_thresh_pyr[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final)]
        results =results+[neuron[2].stimulate.remote(time_array=time_array, amp_array=amp_array_sin, scale1=activ_thresh_pv[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final)]
        results =results+[neuron[3].stimulate.remote(time_array=time_array, amp_array=amp_array_ti, scale1=activ_thresh_pv[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final)]
        results = ray.get(results)
        results = [result[1]/(total_time*1e-03) for result in results]

        fr_activ_pyr[l,0], fr_activ_pyr[l,1], fr_activ_pyr[l,2], fr_activ_pyr[l,3] = fr_rate_sin_pyr[idx_activ_pyr], fr_rate_ti_pyr[idx_activ_pyr], results[0], results[1]
        fr_activ_pv[l,0], fr_activ_pv[l,1], fr_activ_pv[l,2], fr_activ_pv[l,3] = results[2], results[3], fr_rate_sin_pv[idx_activ_pv], fr_rate_ti_pv[idx_activ_pv]
        print("Activation Threshold Metrics Calculated for Direction %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    time_taken_round = time_taken_round*(l)/(l+1)+(time.time()-round_start_time)/(l+1)
    ETA = ((len(unit_vec)-l-1)*time_taken_round)/3600
    print("Simulation Finished for Direction %d! Time Taken %s hr. ETA for script to finish: %s hr"%(l, str(round((time.time()-round_start_time)/3600,3)),str(round(ETA,3))))

if not LOAD_DATA_FLAG:
    #### Saving Processed DATA
    np.save(os.path.join(SAVE_PATH, 'activation_Pyr.npy'), activ_thresh_pyr)
    np.save(os.path.join(SAVE_PATH, 'activation_fr_Pyr.npy'), fr_activ_pyr)
    np.save(os.path.join(SAVE_PATH, 'activation_PV.npy'), activ_thresh_pv)
    np.save(os.path.join(SAVE_PATH, 'activation_fr_PV.npy'), fr_activ_pv)
else:
    fr_activ_pyr = np.load(os.path.join(SAVE_PATH, 'activation_fr_Pyr.npy'))
    fr_activ_pv = np.load(os.path.join(SAVE_PATH, 'activation_fr_PV.npy'))

#### Plotting Activation Threshold
###################################################################################
###################################################################################

### Single Cell Pyramidal Neuron Do not show TI
plt.plot(np.arange(len(unit_vec))+1,fr_activ_pyr[:,0], 'x', label='Pure')
plt.plot(np.arange(len(unit_vec))+1,fr_activ_pyr[:,1], 'x', label='Modulated')
plt.title('Firing Rates of Pyr \n at activation threshold', fontsize=22)
plt.xlabel("Different Electrode Locations", fontsize=20)
plt.ylabel("Firing Rate (Hz)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Pyr_activ_fr.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

### Single Cell PV Neuron n Do not show TI
plt.plot(np.arange(len(unit_vec))+1,fr_activ_pv[:,2], 'x', label='Pure')
plt.plot(np.arange(len(unit_vec))+1,fr_activ_pv[:,3], 'x', label='Modulated')
plt.title('Firing Rates of PV \n at activation threshold', fontsize=22)
plt.xlabel("Different Electrode Locations", fontsize=20)
plt.ylabel("Firing Rate (Hz)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"PV_activ_fr.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

labels = ['Pure-PV','Mod.-PV','Mod.-Pyr']
data = np.hstack([fr_activ_pv[:,2].reshape(-1,1), fr_activ_pv[:,3].reshape(-1,1), fr_activ_pyr[:,1].reshape(-1,1)])
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
plt.boxplot(data, labels=labels)
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.title("Firing Rates of PV and Pyr\n at respective Activation Threshold", fontsize=22)
plt.ylabel("Firing Rate (Hz)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"FiringRate_Activation_Threshold.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

labels = ['PV', 'Pyr']
data = np.hstack([activ_thresh_pv.reshape(-1,1), activ_thresh_pyr.reshape(-1,1)])*10**(-3)
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
plt.boxplot(data, labels=labels)
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.title("Amplitude of Injected Current\n at respective Activation Threshold", fontsize=22)
plt.ylabel("Amplitude Injected Current (mA)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Activation_Threshold.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

labels = ['Pure-Pyr','Mod.-Pyr','Pure-PV','Mod.-PV']
data = fr_activ_pyr
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
plt.boxplot(data, labels=labels)
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.title("Firing Rates of PV and Pyr\n at Pyr Activation Threshold", fontsize=22)
plt.ylabel("Firing Rate (Hz)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"FiringRate_Activation_Pyr_Threshold.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


labels = ['Pure-Pyr','Mod.-Pyr','Pure-PV','Mod.-PV']
data = fr_activ_pv
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
plt.boxplot(data, labels=labels)
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.title("Firing Rates of PV and Pyr\n at PV Activation Threshold", fontsize=22)
plt.ylabel("Firing Rate (Hz)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"FiringRate_Activation_PV_Threshold.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()



labels = ['Pure','Modulated']
data = np.hstack([fr_activ_pyr[:,2].reshape(-1,1)-fr_activ_pyr[:,0].reshape(-1,1),fr_activ_pyr[:,3].reshape(-1,1)-fr_activ_pyr[:,1].reshape(-1,1)]) 
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
plt.boxplot(data, labels=labels)
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
plt.title("Diff. of Firing Rates of PV and Pyr\n at Pyr Activation Threshold", fontsize=22)
plt.ylabel("PV Fr. Rate-Pyr Fr. Rate", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Ratio_Pyr_Thresh.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


labels = ['Pure','Modulated']
data = np.hstack([fr_activ_pyr[:,2].reshape(-1,1)-fr_activ_pyr[:,0].reshape(-1,1),fr_activ_pyr[:,3].reshape(-1,1)-fr_activ_pyr[:,1].reshape(-1,1)]) 
data = np.median(data, axis=0)
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
bar_container = plt.bar(labels, data)
plt.bar_label(bar_container, fmt=lambda x: f'{x:.1f} Hz')
plt.title("Diff. of Firing Rates of PV and Pyr\n at Pyr Activation Threshold", fontsize=22)
plt.ylabel("PV Fr. Rate-Pyr Fr. Rate", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"Ratio_Pyr_Thresh_Bar.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
