import numpy as np
import matplotlib.pyplot as plt

class PulseTrain_Sinusoid:
    
    def amp_train(self, amp, freq, total_time, sampling_rate=1e6):
        
        self.sampling_rate = sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = amp*np.sin(2*np.pi*freq*time_array*1e-3)
        
        #ramp_samples = np.sum(time_array<200)
        #ramp_inc = np.linspace(0,amp,ramp_samples)
        #ramp_dec = np.flip(ramp_inc.copy())
        #amp_array[:ramp_samples] = amp_array[:ramp_samples]*ramp_inc
        #amp_array[-ramp_samples:] = amp_array[-ramp_samples:]*ramp_dec
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()

class PulseTrain_square:
    def amp_train(self,amp,pulse_width,period,total_time,sampling_rate):
        self.sampling_rate = sampling_rate
        total_samples = total_time * sampling_rate * 1e-3 # total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        #num samples in one pulse, in one period
        num_samples_pulse = int(pulse_width * sampling_rate * 1e-3)
        num_samples_period = int(period * sampling_rate * 1e-3)
        amp_array = np.zeros(int(total_samples))
        for i in range(0, int(total_samples), num_samples_period):
            end_index = min(i + num_samples_pulse, total_samples)
            amp_array[i:end_index] = amp
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()


class PulseTrain_TI:
    
    def amp_train(self, amp1, amp2, freq1, freq2, total_time, sampling_rate=1e6):
        
        self.sampling_rate = sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = amp1*np.sin(2*np.pi*freq1*time_array*1e-3)+amp2*np.sin(2*np.pi*freq2*time_array*1e-3)
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()
    
