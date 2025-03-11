import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from neuron_model_serial import NeuronSim
from elec_field import ICMS
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI
import sys
from helper import fibonacci_sphere, plot_electrode_and_neuron

##################################################################################
################## ICMS Monopolar Experimental Setup #############################
##################################################################################
SEED = 1234
np.random.seed(SEED)
print(f"Setting Random Seed as {SEED}")
cwd = os.getcwd()
print(f"Working in the directory: {cwd}. All data will be saved and loaded relative to this directory")

#### Defining Variables for Setting up Simulation
cell_id_pyr_lst = [6, 7, 8, 9, 10]  # Different Morphology for L23 Pyr Cells
cell_id_pv_lst = [32, 33, 34, 35, 36]  # Different Morphology for L23 LBC Cells
human_or_mice = 1  # 1 -> mice, 0 -> human
temp = 34.0  # Celsius
dt = 0.025  # ms
num_cores = 1  # Number of Cores used for Parallelization
SHOW_PLOTS = False  # Flag used for showing or not showing plots
dist = float(sys.argv[1])  # mm, distance from the origin for the ICMS electrode
elec_location_ICMS = fibonacci_sphere(samples=30)  # Sampling 30 approximately uniformly spaced electrode locations from a unit sphere
elec_location_ICMS *= dist  # Scaling the radius of the sphere to the dist variable

angle_pv = np.array([0, 0])  # parameter used for specifying rotation of PV morphology
angle_pyr = np.array([0, 0])  # parameter used for specifying rotation of Pyr morphology

loc_pyr = np.array([0, 0, 0])  # parameter used for specifying location of Pyr morphology
loc_pv = np.array([0, 0, 0])  # parameter used for specifying location of PV morphology

SAVE_PATH = os.path.join(os.getcwd(), f'TISimResults/PointElectrodeSim/Results_distance{int(dist*1000)}um')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

PLOT_NEURON_AND_ELECTRODE = True
if PLOT_NEURON_AND_ELECTRODE:
    # Plot Pyr Coordinates
    for cell_id_pyr in cell_id_pyr_lst:
        # Get Neuron Coordinates
        neuron = NeuronSim(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt)
        coord = neuron._translate_rotate_neuron(pos_neuron=loc_pyr, angle=angle_pyr)

        # Plot ICMS Electrode With Neuron
        savepath_curr = os.path.join(SAVE_PATH, f'NeuronOrientation_ElecLocation{int(dist*1000)}um_cellid{cell_id_pyr}.png')
        plot_electrode_and_neuron(coord_elec=elec_location_ICMS * 1000, coord=coord, savepath=savepath_curr)

    # Plot PV Coordinates
    for cell_id_pv in cell_id_pv_lst:
        # Get Neuron Coordinates
        neuron = NeuronSim(human_or_mice=human_or_mice, cell_id=cell_id_pv, temp=temp, dt=dt)
        coord = neuron._translate_rotate_neuron(pos_neuron=loc_pyr, angle=angle_pv)

        # Plot ICMS Electrode With Neuron
        savepath_curr = os.path.join(SAVE_PATH, f'NeuronOrientation_ElecLocation_cellid{cell_id_pv}.png')
        plot_electrode_and_neuron(coord_elec=elec_location_ICMS * 1000, coord=coord, savepath=savepath_curr)

###################################################################################
#### Monopolar Stimulation
###################################################################################
sim_already_performed = len(os.listdir(SAVE_PATH))
print(f"Simulation Already Performed for {sim_already_performed} Electrode Locations. Starting the Simulation from Electrode Location {sim_already_performed+1}")

# Generate Waveforms
print("Generating Waveform...")
pulse_train_sin = PulseTrain_Sinusoid()
pulse_train_ti = PulseTrain_TI()
freq1, freq2 = 2 * 1e3, 2.02 * 1e3  # Hz
total_time, sampling_rate = 2000, 1e6  # ms, Hz
amp_array_sin, time_array = pulse_train_sin.amp_train(amp=1, freq=freq1, total_time=total_time, sampling_rate=sampling_rate)
amp_array_ti, _ = pulse_train_ti.amp_train(amp1=0.5, amp2=0.5, freq1=freq1, freq2=freq2, total_time=total_time, sampling_rate=sampling_rate)

print("Waveform Generated!")

activ_thresh_pyr = np.empty(len(elec_location_ICMS))
activ_thresh_pv = np.empty(len(elec_location_ICMS))
fr_activ_pyr = np.empty([len(elec_location_ICMS), 4])
fr_activ_pv = np.empty([len(elec_location_ICMS), 4])

# Stimulation for each electrode location
for l, elec_loc in enumerate(elec_location_ICMS):
    print(f"Starting Simulation for Electrode Location {l + 1}")
    # Generate Electric Field Simulator
    elec_field = ICMS(x=elec_loc[0], y=elec_loc[1], z=elec_loc[2], conductivity=0.33)

    # Run Pyr Stimulation
    cell_id_pyr = cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst))]
    neuron = NeuronSim(human_or_mice=human_or_mice, cell_id=cell_id_pyr, temp=temp, dt=dt, elec_field=elec_field)
    neuron._set_xtra_param(angle=angle_pyr, pos_neuron=loc_pyr)
    result = neuron.stimulate(time_array=time_array, amp_array=amp_array_ti, scale1=1, sampling_rate=sampling_rate)
    print(f"Simulation result: {result}")

    # Add more simulation logic as needed...

print("Simulation Complete!")
