import numpy as np
import matplotlib.pyplot as plt
import os
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI

####################################################################################
###################### Plotting Sinusoidal Waveforms ###############################
####################################################################################
pulse_train_sin = PulseTrain_Sinusoid()
units = 'mA'
amp = 1 ## a.u.
freq = 2*1e3 ## Hz
sampling_rate = 1e06
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/InputWaveforms')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

### Zoomed-In-waveform
total_time = 200 ##ms
pulse_train_sin.amp_train(amp=amp, freq=freq, total_time=total_time, sampling_rate=sampling_rate)
pulse_train_sin.plot_waveform(save_path=os.path.join(SAVE_PATH,'Sinusoid_Frequency'+str(int(freq))+"Hz_"+str(total_time)+"ms.png"), units=units, quantity='Injected Current', show=True)

### Total Waveform 
total_time = 2000 ##ms
pulse_train_sin.amp_train(amp=amp, freq=freq, total_time=total_time, sampling_rate=sampling_rate)
pulse_train_sin.plot_waveform(save_path=os.path.join(SAVE_PATH,'Sinusoid_Frequency'+str(int(freq))+"Hz_"+str(total_time)+"ms.png"), units=units, quantity='Injected Current', show=True)

####################################################################################
######################### Plotting TI Waveforms ####################################
####################################################################################
pulse_train_TI = PulseTrain_TI()
amp1 = 0.5 ## a.u.
amp2 = 0.5 ## a.u.
freq1 = 2*1e3 ## Hz
freq2 = 2.02*1e3 ## Hz
sampling_rate = 1e06
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/InputWaveforms')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

### Zoomed-In-waveform
total_time = 200 ##ms
pulse_train_TI.amp_train(amp1=amp1,amp2=amp2, freq1=freq1,freq2=freq2, total_time=total_time, sampling_rate=sampling_rate)
pulse_train_TI.plot_waveform(save_path=os.path.join(SAVE_PATH,'TI_CarrierFreq'+str(int(freq1))+"Hz_BeatFreq"+str(int(freq2-freq1))+"Hz_"+str(total_time)+"ms.png"), units=units, quantity='Injected Current', show=True)

### Total Waveform 
total_time = 2000 ##ms
pulse_train_TI.amp_train(amp1=amp1,amp2=amp2, freq1=freq1,freq2=freq2, total_time=total_time, sampling_rate=sampling_rate)
pulse_train_TI.plot_waveform(save_path=os.path.join(SAVE_PATH,'TI_CarrierFreq'+str(int(freq1))+"Hz_BeatFreq"+str(int(freq2-freq1))+"Hz_"+str(total_time)+"ms.png"), units=units, quantity='Injected Current', show=True)


