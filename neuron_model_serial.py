import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm

import os
import time
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression as lin_fit

import plotly
import logging

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'

class NeuronSim:
    
    def __init__(self, human_or_mice=0, cell_id=None, temp=37, dt=0.025, show_morphology_old=False, elec_field=None, elec_field2=None):
        
        cwd = os.getcwd()##
        from neuron import h, gui, rxd
        from neuron.units import ms, mV
        from hoc2swc import neuron2swc
       
        ############################ Loading Neuron .hoc files to use Neuron #############################
        h.load_file("stdrun.hoc")
        h.load_file("nrngui.hoc")
        h.load_file("import3d.hoc")
        h.load_file("nrngui.hoc")
        h.load_file("interpCoordinates.hoc")
        h.load_file("setPointers.hoc")
        h.load_file("cellChooser.hoc")
        h.load_file("setParams.hoc")
        h.load_file("editMorphology.hoc")
        if show_morphology_old:
            h.load_file("calcVe.hoc")
            h.load_file('color_max.hoc')
            h('color_plotmax()')
        
        class read_cell:

            def __init__(self, human_or_mice=0, cell_id=None, temp=37, dt=0.025, show_morphology_old=False):
                cwd = os.getcwd()
                if human_or_mice is None:
                    human_or_mice = int(input("Input 0 for parameters to be set for human neuron or 1 for rat neuron:"))
                self.human_or_mice = human_or_mice
                if self.human_or_mice==0:
                    h('setParamsAdultHuman()')
                else:
                    h('setParamsAdultRat()')
                
                self.cell_dict  =["L1_NGC-DA_bNAC219_1","L1_NGC-DA_bNAC219_2","L1_NGC-DA_bNAC219_3","L1_NGC-DA_bNAC219_4","L1_NGC-DA_bNAC219_5",\
                            "L23_PC_cADpyr229_1","L23_PC_cADpyr229_2","L23_PC_cADpyr229_3","L23_PC_cADpyr229_4","L23_PC_cADpyr229_5",\
                            "L4_LBC_cACint209_1","L4_LBC_cACint209_2","L4_LBC_cACint209_3","L4_LBC_cACint209_4","L4_LBC_cACint209_5",\
                            "L5_TTPC2_cADpyr232_1","L5_TTPC2_cADpyr232_2","L5_TTPC2_cADpyr232_3","L5_TTPC2_cADpyr232_4","L5_TTPC2_cADpyr232_5",\
                            "L6_TPC_L4_cADpyr231_1","L6_TPC_L4_cADpyr231_2","L6_TPC_L4_cADpyr231_3","L6_TPC_L4_cADpyr231_4","L6_TPC_L4_cADpyr231_5","L23_ChC_cACint209_1",\
                            "L23_ChC_cACint209_2","L23_ChC_cACint209_3","L23_ChC_cACint209_4","L23_ChC_cACint209_5","L23_LBC_bNAC219_1","L23_LBC_cNAC187_1","L23_LBC_cNAC187_2",\
                            "L23_LBC_cNAC187_3","L23_LBC_cNAC187_4","L23_LBC_cNAC187_5","L5_TTPC1_cADpyr232_1","L5_TTPC1_cADpyr232_2","L5_TTPC1_cADpyr232_3","L5_TTPC1_cADpyr232_4",\
"L5_TTPC1_cADpyr232_5"]
                
                if cell_id is None:
                    for i in range(len(cell_dict)):
                        print("Choose cell id  %d for the cell model %s"%(i+1,cell_dict[i]))
                    cell_id = int(input("Input the desired cell id:"))
                self.cell_id = cell_id
                h('cell_chooser(%s)'%str(cell_id))
                h.celsius = temp
                h.dt = dt
                print("Temparature chosen for simulation:", h.celsius)
                print("Discretization Step for simulation: %s ms"%(str(h.dt)))
       
            def _set_extracellular_stim(self):
                h.load_file(cwd+"/Backend_Code/fixnseg.hoc")
                h('geom_nseg()')

                for sec in h.allsec():
                    if sec.nseg == 1:
                        sec.nseg == 3
                    sec.insert('extracellular')
                    sec.insert('xtra')
                 
                h.load_file(cwd + "/Backend_Code/Extracellular_Stim/interpxyz.hoc")
                h.load_file(cwd + "/Backend_Code/Extracellular_Stim/setpointers.hoc")

            def _reset_temp_dt(self, temp=None, dt=None):
                if temp is not None:
                    h.celsius = temp
                else:
                    h.celsius = 34
                if dt is None:
                    h.dt = 0.025
                else:
                    h.dt =dt

            def _cart_to_sph(self, pos):
                if len(pos.shape) == 1:
                    pos = pos.reshape(1,-1)
                r = np.sqrt(np.sum(pos**2, axis=1)).reshape(1,-1)
                theta = np.arccos(pos[:,2]/r).reshape(1,-1)
                phi = np.arctan2(pos[:,1],pos[:,0]).reshape(1,-1)
                sph_pos = np.hstack([r,theta,phi])
                return sph_pos
            
            def _sph_to_cart(self, pos):
                if len(pos.shape) == 1:
                    pos = pos.reshape(1,-1)
                x = pos[:,0]*np.sin(pos[:,1])*np.cos(pos[:,2])
                y = pos[:,0]*np.sin(pos[:,1])*np.sin(pos[:,2])
                z = pos[:,0]*np.cos(pos[:,1])
                cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
                return cart_pos

            def plot_neuron_default(self):
                ps= h.PlotShape(False)
                ps.plot(plotly).show()
            
            def plot_neuron_better(self, sec_xyz):
                sec_xyz = np.vstack(sec_xyz)
                view_angle = np.linspace(0,360,361)
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                
                img = ax.scatter(sec_xyz[:,0], sec_xyz[:,1], sec_xyz[:,2], linewidth=0.3, alpha=1.0, c='blue')
                ax.set_xlabel('X-axis (um)', fontsize=16)
                ax.set_ylabel('Y-axis (um)', fontsize=16)
                ax.set_zlabel('Z-axis (um)', fontsize=16)
                ax.set_title('Neuron Orientation', fontsize=21)
                ax.tick_params(axis='x',labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.tick_params(axis='z',labelsize=12)
                plt.tight_layout()
                
                def update(frame):
                    ax.view_init(10,view_angle[frame])
                ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
                #ani.save(os.path.join(neuron_orient_dir,'NeuronOrientation_cellid'+str(cell_id)+'_layer5.gif'), writer='pillow')
                plt.show()
                ## Get the 3-d points of the main axon
            
            def _get_reference_loc(self):
                coord_3d_main_axon = []
                for sec in h.main_ax_list:
                    coord_3d_main_axon.append([sec(0.5).xtra.x, sec(0.5).xtra.y, sec(0.5).xtra.z])
                coord_3d_main_axon = np.array(coord_3d_main_axon)
                #print(1,coord_3d_main_axon.shape)
                uu, dd, vv = np.linalg.svd(coord_3d_main_axon - np.mean(coord_3d_main_axon, axis=0))
                axon_dir = vv[0]/np.sqrt(np.sum(vv[0]**2)) 
                axon_dir_sph = self._cart_to_sph(axon_dir)
                soma = list(h.allsec())[0]
                soma_loc = np.array([soma(0.5).xtra.x,soma(0.5).xtra.y,soma(0.5).xtra.z])
                print("The reference position being used to orient neuron is x: %s um, y: %s um, z %s um "%(str(round(soma_loc[0],3)),str(round(soma_loc[1],3)),str(round(soma_loc[2],3))))
                print("The roughly linear direction of the main axon is the unit direction : [%s, %s, %s] "%(str(round(axon_dir[0],3)),str(round(axon_dir[1],3)),str(round(axon_dir[2],3))))
                #self.plot_neuron_shape()
                return axon_dir.flatten(), axon_dir_sph.flatten(), soma_loc.flatten(), coord_3d_main_axon
    
       
            def _translate_rotate_neuron(self, pos_neuron=np.array([0,0,0]), angle=np.array([0,0]), plot=False):
                axon_dir, axon_sph_dir, soma_loc, main_axon_coord = self._get_reference_loc()
                current_theta, current_phi = axon_sph_dir[1], axon_sph_dir[2]

                sec_xyz_tr = []
                sec_lst = []
                for sec in h.allsec():
                    if True:#h.ismembrane('xtra', sec=sec):
                        seg_xyz_tr = []
                        for seg in sec:
                            x, y, z =seg.xtra.x, seg.xtra.y, seg.xtra.z
                            
                            ## Translating the coordinate system to make sure that soma is at origin
                            x_tr, y_tr, z_tr = x-soma_loc[0], y-soma_loc[1], z-soma_loc[2]             
                            
                            ## Performinig phi-rotation which is equivalent to x-y rotation to align the main axon in the x-z plane
                            des_phi_rot =0-current_phi
                            x_tmp, y_tmp = x_tr*np.cos(des_phi_rot)-y_tr*np.sin(des_phi_rot),x_tr*np.sin(des_phi_rot)+y_tr*np.cos(des_phi_rot) 
                            x_tr, y_tr = x_tmp, y_tmp 
                            
                            ## Performinig theta-rotation which is now equivalent to z-x rotation to align the main axon z-axis 
                            des_theta_rot = -1*(np.pi-current_theta)
                            z_tmp, x_tmp = z_tr*np.cos(des_theta_rot)-x_tr*np.sin(des_theta_rot),z_tr*np.sin(des_theta_rot)+x_tr*np.cos(des_theta_rot) 
                            x_tr, z_tr = x_tmp, z_tmp
                            seg_xyz_tr.append([x_tr,y_tr,z_tr])
                        sec_xyz_tr.append(seg_xyz_tr)
                        sec_lst.append(sec)
                
                sec_xyz_tr_plot = []
                for (sec, sec_coord) in zip(sec_lst, sec_xyz_tr):
                    seg_xyz_tr_plot = []
                    for (seg,seg_coord) in zip(sec, sec_coord):
                        x, y, z = seg_coord[0], seg_coord[1], seg_coord[2]
                        ## theta rotation
                        z_tmp, x_tmp = z*np.cos(angle[0])-x*np.sin(angle[0]),z*np.sin(angle[0])+x*np.cos(angle[0]) 
                        x, z = x_tmp, z_tmp
                        ## phi rotation  
                        x_tmp, y_tmp = x*np.cos(angle[1])-y*np.sin(angle[1]),x*np.sin(angle[1])+y*np.cos(angle[1]) 
                        x, y = x_tmp, y_tmp
                        ## translation to pos_neuron
                        x, y, z = x+pos_neuron[0], y+pos_neuron[1], z+pos_neuron[2]
                        seg_xyz_tr_plot.append([x,y,z])
                    sec_xyz_tr_plot.append(seg_xyz_tr_plot)
                sec_xyz_tr_plot = np.vstack(sec_xyz_tr_plot)
                return sec_xyz_tr_plot

            def _set_xtra_param(self, elec_field=None, elec_field2=None, pos_neuron=np.array([0,0,0]), angle=np.array([0,0]), debug=False):
                ### Orienting the neuron along the z-axis with soma at origin and the main axon pointing in the negative z-axis 
                axon_dir, axon_sph_dir, soma_loc, main_axon_coord = self._get_reference_loc()
                current_theta, current_phi = axon_sph_dir[1], axon_sph_dir[2]
                sec_xyz_tr = []
                sec_lst = []
                for sec in h.allsec():
                    if h.ismembrane('xtra', sec=sec):
                        seg_xyz_tr = []
                        for seg in sec:
                            x, y, z =seg.xtra.x, seg.xtra.y, seg.xtra.z            
                            ## Translating the coordinate system to make sure that soma is at origin
                            x_tr, y_tr, z_tr = x-soma_loc[0], y-soma_loc[1], z-soma_loc[2]             
                            ## Performinig phi-rotation which is equivalent to x-y rotation to align the main axon in the x-z plane
                            des_phi_rot =0-current_phi
                            x_tmp, y_tmp = x_tr*np.cos(des_phi_rot)-y_tr*np.sin(des_phi_rot),x_tr*np.sin(des_phi_rot)+y_tr*np.cos(des_phi_rot) 
                            x_tr, y_tr = x_tmp, y_tmp                             
                            ## Performinig theta-rotation which is now equivalent to z-x rotation to align the main axon z-axis 
                            des_theta_rot = -1*(np.pi-current_theta)
                            z_tmp, x_tmp = z_tr*np.cos(des_theta_rot)-x_tr*np.sin(des_theta_rot),z_tr*np.sin(des_theta_rot)+x_tr*np.cos(des_theta_rot) 
                            x_tr, z_tr = x_tmp, z_tmp
                            seg_xyz_tr.append([x_tr,y_tr,z_tr])
                        sec_xyz_tr.append(seg_xyz_tr)
                        sec_lst.append(sec)
                
                for (sec, sec_coord) in zip(sec_lst, sec_xyz_tr):
                    for (seg,seg_coord) in zip(sec, sec_coord):
                        x, y, z = seg_coord[0], seg_coord[1], seg_coord[2]
                        ## theta rotation
                        z_tmp, x_tmp = z*np.cos(angle[0])-x*np.sin(angle[0]),z*np.sin(angle[0])+x*np.cos(angle[0]) 
                        x, z = x_tmp, z_tmp
                        ## phi rotation  
                        x_tmp, y_tmp = x*np.cos(angle[1])-y*np.sin(angle[1]),x*np.sin(angle[1])+y*np.cos(angle[1]) 
                        x, y = x_tmp, y_tmp
                        ## translation to pos_neuron
                        x, y, z = x+pos_neuron[0], y+pos_neuron[1], z+pos_neuron[2]
                        seg.xtra.es = elec_field.eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)) ## um in mm
                        if elec_field2 is not None:
                            seg.xtra.es2 = elec_field2.eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)) ## um in mm
                        else:
                            seg.xtra.es2 = 0
             
             
      
            def stimulate(self, time_array, amp_array, amp_array2=None, sampling_rate=1e5, delay_init=2, delay_final=2, plot=False, Vinit=-65, time_stamps=None, save_state_show=False):
                cwd = os.getcwd()
                save_state = os.path.join('cells/SaveState')
                if not os.path.exists(save_state):
                    os.makedirs(save_state)
                delay_final, delay_init = int(delay_final), int(delay_init)
                save_state = os.path.join(save_state,'human_or_mice'+str(self.human_or_mice)+'cell-'+str(self.cell_id)+'_Temp-'+str(h.celsius)+'C_dt-'+str(h.dt*10**3)+'us_delay-'+str(delay_init)+'ms.bin')
                if not os.path.exists(save_state):
                    print("Creating Save State for the neuron type ....")
                    burn_in = delay_init  # ms 
                    burn_in_sample = np.linspace(0, burn_in, int(sampling_rate * burn_in * 1e-3))
                    burn_in_amp = np.zeros(len(burn_in_sample))
                
                
                    amp_array_ss = burn_in_amp.copy()            
                    time_array_ss = burn_in_sample.copy()            
                    t_vec = h.Vector(time_array_ss)
                    stim_waveform_ss = h.Vector(amp_array_ss) 
                    stim_waveform_ss.play(h._ref_stim_xtra, t_vec, 1)
                 
                    amp_array_ss_2 = burn_in_amp.copy()            
                    time_array_ss_2 = burn_in_sample.copy()            
                    t_vec_2 = h.Vector(time_array_ss_2)
                    stim_waveform_ss_2 = h.Vector(amp_array_ss_2) 
                    stim_waveform_ss_2.play(h._ref_stim2_xtra, t_vec, 1)   
                    
                    soma = list(h.allsec())[0]
                    
                    soma_recording = h.Vector().record(soma(0.5)._ref_v) ## Spikes are recorded in the soma
                    t = h.Vector().record(h._ref_t) 
                    
                    h.finitialize(Vinit * mV)
                    h.continuerun((np.max(time_array_ss)) * ms)
                    ss = h.SaveState()
                    ss.save()
                    sf = h.File(save_state)
                    ss.fwrite(sf)
                    self.soma_recording = np.array(soma_recording)
                    self.t = np.array(t)

                    plt.plot(self.t[self.t>0], self.soma_recording[self.t>0])
                    plt.title("Soma Membrane Potential", fontsize='22')
                    plt.xlabel("Time (ms)", fontsize=20)
                    plt.ylabel("Membrane Potential (mV)", fontsize=20)
                    plt.xticks(fontsize=18)
                    plt.yticks(fontsize=18)
                    plt.tight_layout()
                    if save_state_show:
                        plt.show()
                    else:
                        plt.close()
                    print("Finished Creating the Save State for the neuron type")
                    return True 
                
                ## burn_in_period
                burn_in = delay_init+2  # ms
                burn_out = delay_final  # ms
                
                burn_in_sample = np.linspace(0, burn_in, int(sampling_rate * burn_in * 1e-3))
                burn_in_amp = np.zeros(len(burn_in_sample))
                
                burn_out_sample = np.linspace(0, burn_out, int(sampling_rate * burn_out * 1e-3))
                burn_out_amp = np.zeros(len(burn_out_sample))
                
                amp_array = np.hstack((burn_in_amp, amp_array, burn_out_amp))
                if amp_array2 is not None:
                    amp_array2 = np.hstack((burn_in_amp.copy(), amp_array2, burn_out_amp))
                else:
                    amp_array2 = np.zeros(amp_array.shape)

                time_array_tmp = np.hstack((burn_in_sample, time_array + burn_in, burn_out_sample + time_array[len(time_array) - 1] + burn_in))
                time_array = time_array_tmp.copy()
                time_array_2 = time_array_tmp.copy() 
                t_vec = h.Vector(time_array)
                stim_waveform = h.Vector(amp_array)
                t_vec_2 = h.Vector(time_array_2)
                stim_waveform_2 = h.Vector(amp_array2)
                
		## Extracellular 
                stim_waveform.play(h._ref_stim_xtra, t_vec, 1)
                stim_waveform_2.play(h._ref_stim2_xtra, t_vec_2, 1)

                ## Choose Recording Site
                soma = list(h.allsec())[0]
                soma_recording = h.Vector().record(soma(0.5)._ref_v) ## Spikes are recorded in the soma
                           
                ## Record Time
                t = h.Vector().record(h._ref_t)                 
                h.finitialize(Vinit * mV)
                
                ns = h.SaveState()
                sf_new = h.File(save_state)
                ns.fread(sf_new)
                ns.restore(1)

                h.continuerun((np.max(time_array)) * ms)
                
                self.soma_recording = np.array(soma_recording)
                self.t = np.array(t)
                peaks, _ = find_peaks(self.soma_recording, prominence=40)
                idx = self.soma_recording[peaks] > -25
                
                peaks = peaks[idx]
                self.peaks = peaks.copy()
                t_peaks = self.t[peaks].copy()
                
                if time_stamps is not None:
                    self.peaks_lst = []
                    for i in range(len(time_stamps)-1):
                        t_peaks_copy = t_peaks.copy()
                        idx_left= t_peaks_copy>time_stamps[i]
                        peaks_btw_time_stamps = peaks[idx_left]
                        t_peaks_copy  = t_peaks_copy[idx_left]
                        idx_right = t_peaks_copy<time_stamps[i+1]
                        peaks_btw_time_stamps = peaks_btw_time_stamps[idx_right]
                        self.peaks_lst.append(peaks_btw_time_stamps)
                
                if time_stamps is None:
                    return self.soma_recording, len(self.peaks), self.t
                else:
                    return self.soma_recording, np.array([len(peak) for peak in self.peaks_lst]).flatten(), self.t
            
            def save_soma_recording(self, delay_init=1000):
                soma_recording = self.soma_recording[self.t>delay_init]
                t = self.t[self.t>delay_init]-delay_init
                return soma_recording, t
   
            def plot_sim_result(self, save_path=None, show=True, delay_init=1000):
                plt.plot(self.t[self.t>delay_init]-delay_init, self.soma_recording[self.t>delay_init])
                plt.plot(self.t[self.peaks]-delay_init, self.soma_recording[self.peaks], 'x')
                plt.title("Soma Membrane Potential", fontsize='22')
                plt.xlabel("Time (ms)", fontsize=20)
                plt.ylabel("Membrane Potential (mV)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.tight_layout()
                if save_path is not None:
                    plt.savefig(save_path)
                if show:
                    plt.show()
                else:
                    plt.clf()
                    plt.cla()
                    
        self.human_or_mice=human_or_mice
        self.cell_id=cell_id
        self.temp=temp
        self.dt=dt
        self.show_morphology_old=show_morphology_old       
        self.elec_field = elec_field    
        self.elec_field2=elec_field2
        self.cell = read_cell(human_or_mice=self.human_or_mice, cell_id=self.cell_id, temp=self.temp, dt=self.dt, show_morphology_old=self.show_morphology_old)

    def stimulate(self, time_array, amp_array, amp_array2=None, scale1=1, scale2=1, sampling_rate=1e5, delay_init=2, delay_final=2, plot=False, Vinit=-65, time_stamps=None, save_state_show=False):
        amp_array = scale1*amp_array
        if amp_array2 is not None:
            amp_array2 = scale2*amp_array2
        return self.cell.stimulate(time_array=time_array, amp_array=amp_array, amp_array2=amp_array2, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, plot=plot, Vinit=Vinit, save_state_show=save_state_show) 
    
    def _set_xtra_param(self, pos_neuron=np.array([0,0,0]), angle=np.array([0,0]), debug=False):
        self.cell._set_xtra_param(elec_field=self.elec_field, elec_field2=self.elec_field2,pos_neuron=pos_neuron, angle=angle, debug=debug) 
    
    def _translate_rotate_neuron(self, pos_neuron=np.array([0,0,0]), angle=np.array([0,0]), plot=False):
        return self.cell._translate_rotate_neuron(pos_neuron=pos_neuron, angle=angle, plot=plot)
        
    def _reset_elec_field(self, elec_field):
        self.elec_field = elec_field
        self.cell.elec_field = elec_field
                
    def plot_sim_result(self, save_path=None, show=True, delay_init=1000):
        self.cell.plot_sim_result(save_path=save_path, show=show, delay_init=delay_init)

    def save_soma_recording(self, delay_init=1000):
        return self.cell.save_soma_recording(delay_init=delay_init)
