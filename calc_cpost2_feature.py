import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py

def moving_average_filter(pulse, window_size):
    window = np.ones(window_size) / float(window_size)
    smooth_pulse = np.convolve(pulse, window, mode='same')
    return smooth_pulse

def calc_trigger_time(pulse, time, arrival_time, window_size):
    pre_event_point = np.where(time < arrival_time)
    trigger_time = []
    for j in range(len(pulse)):
        # Pre-event average voltage for trigger calculation
        pulse_average_before_event = np.average(pulse[j][pre_event_point])
        pulse_trigger = -pulse_average_before_event + 0.05
        
        # Smoothing
        smooth_pulse = moving_average_filter(-pulse[j], window_size)
        
        # Calculate trigger time
        trigger_point = np.where((smooth_pulse[1:] > pulse_trigger) & (smooth_pulse[0:-1] < pulse_trigger))[0]
        if len(trigger_point) != 0:
            trigger_time.append(time[trigger_point][0])
    
    return trigger_time 

def process_file(file_path, ch, j):
    with h5py.File(file_path, 'r') as file:
        hres = file['waveform']['hres'][()]
        vres = file['waveform']['vres'][()]
        pulse = np.array(file['waveform']['pulse'])

    pulse = pulse * vres 
    time = np.arange(len(pulse[0])) * hres
    area = np.sum(-pulse * hres,axis = -1)
    pulse_heights = np.min(pulse, axis=-1)
    pulse_heights_indice = np.argmin(pulse, axis=-1)
    peak_times = pulse_heights_indice * hres

    trigger_times = []
    for k in range(0, 5, 1):
        pulse_n = pulse[1000 * k : 1000 * (k + 1)]
        trigger_time = calc_trigger_time(pulse_n, time, 0.00002, 25)
        if trigger_time is not None:
            trigger_times.append(trigger_time)
    trigger_times = np.concatenate(trigger_times)

    data_to_save = {
        'trigger_time': trigger_times,
        'pulse_height': pulse_heights,
        'peak_times': peak_times,
        'area': area,
    }

    print("pulse_heights", len(pulse_heights))
    print("peak_times", len(peak_times))
    print("trigger_times", len(trigger_times))
    print("area", len(area))

    save_path = f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch{ch}_{j}.npz'
    np.savez(save_path, **data_to_save)

for ch in range(1, 4, 2):
    for j in range(1, 11, 1):
        file_path = f'/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch{ch}_{j}.hdf5'
        process_file(file_path, ch, j)


""" import os
import sys
import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as cr
import optparse
import matplotlib.pylab as plab
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import h5py

def moving_average_filter(pulse, window_size):
    window = np.ones(window_size) / float(window_size)
    smooth_pulse = np.convolve(pulse, window, mode='same')
    return smooth_pulse

def calc_trigger_time(pulse, time, arrival_time, window_size):
    pre_event_point = np.where(time < arrival_time)
    trigger_time = []
    for j in range(len(pulse)):
        # 立ち上がり前の電圧でトリガーを計算
        pulse_average_before_event = np.average(pulse[j][pre_event_point])
        pulse_trigger = -pulse_average_before_event + 0.05
        
        # 平滑化
        smooth_pulse = moving_average_filter(-pulse[j], window_size)
        
        # trigger_timeを計算
        trigger_point = np.where((smooth_pulse[1:] > pulse_trigger) & (smooth_pulse[0:-1] < pulse_trigger))[0]
        if len(trigger_point) != 0:
            trigger_time.append(time[trigger_point][0])
    
    return trigger_time 

for i in range(1, 4, 2):
    for j in range(1, 11, 1):
        file_path = f'/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch{i}_{j}.hdf5'
        with h5py.File(file_path, 'r') as file:
            hres = file['waveform']['hres'][()]
            vres = file['waveform']['vres'][()]
            pulse = np.array(file['waveform']['pulse'])

        pulse = pulse * vres 
        time = np.arange(len(pulse[0])) * hres
        area = -pulse * hres
        pulse_heights = np.min(pulse, axis=-1)
        pulse_heights_indice = np.argmin(pulse, axis=-1)
        peak_times = pulse_heights_indice * hres

        trigger_times = []
        for k in range(0, 5, 1):
            pulse_n = pulse[1000 * k : 1000 * (k + 1)]
            trigger_time = calc_trigger_time(pulse_n, time, 0.00002, 25)
            if trigger_time is not None:
                trigger_times.append(trigger_time)
        trigger_times = np.concatenate(trigger_times)

        data_to_save = {
            'trigger_time': trigger_times,
            'pulse_height': pulse_heights,
            'peak_times': peak_times,
            'area': area,
        }

        print("pulse_heights", len(pulse_heights))
        print("peak_times", len(peak_times))
        print("trigger_times", len(trigger_times))

        save_path = f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch{i}_{j}.npz'
        np.savez(save_path, **data_to_save)
 
 """
""" for i in range(1,4,2):
  for j in range(1,11,1):
    file = h5py.File(f'/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch{i}_{j}.hdf5','a')
    hres = file['waveform']['hres'][()]
    vres = file['waveform']['vres'][()]
    pulse = np.array(file['waveform']['pulse'])
    file.close()
    pulse = pulse * vres 
    time = np.arange(len(pulse[0]))*hres
    area = -pulse * hres
    pulse_heights = np.min(pulse,axis = -1)
    pulse_heights_indice = np.argmin(pulse,axis = -1)
    peak_times = pulse_heights_indice*hres
    
    trigger_times = []
    for j in  range(0,5,1):
        pulse_n = pulse[1000*j:1000*(j+1)]
        trigger_time = calc_trigger_time(pulse_n ,time ,0.00002 ,25)
        if trigger_time is not None:
          trigger_times.append(trigger_time)
    trigger_times = np.concatenate(trigger_times)
    
    data_to_save = {
      'trigger_time':trigger_times,
      'pulse_height':pulse_heights,
      'peak_times':peak_times,
      'area':area,
      } 
    
    print("pulse_heights",len(pulse_heights))
    print("peak_times",len(peak_times))
    print("trigger_times",len(trigger_times))

    np.savez(f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch{i}_{j}.npz',**data_to_save) 
      
 """
""" for i in range(1,4,2):
  for j in range(1,11,1):
   

file = h5py.File(f'/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/bnon/cpost02pn_ch1_1.hdf5','a')
hres = file['waveform']['hres'][()]
vres = file['waveform']['vres'][()]
time = file['waveform']['pulse'][0]*hres
trigger_times = []
areas = []
pulse_heights = []
peak_times = []
for k in range(0,5,1):
  pulse = np.array(file['waveform']['pulse'])[1000*k:1000*(k+1)]
pulse = pulse * vres
file.close

area = -pulse * hres
pulse_height = np.min(pulse,axis = -1)
pulse_heights_indice = np.argmin(pulse,axis = -1)
peak_time = pulse_heights_indice*hres
trigger_time = calc_trigger_time(pulse ,time ,0.00002 ,25)

if trigger_time is not None:
  trigger_times.append(trigger_time)
pulse_heights.append(pulse_height)
peak_times.append(peak_time)
areas.append(area)

trigger_times = np.concatenate(trigger_times)
areas = np.concatenate(areas)
pulse_heights = np.concatenate(pulse_heights)
peak_times = np.concatenate(peak_times)

plt.figure()
for i in range(20):
 plt.plot(time,pulse[i])
 plt.axvline(peak_times[i])
 plt.axvline(trigger_times[i])
 plt.axhline(pulse_heights[i])
 plt.show() 


print("pulse_heights: ",len(pulse_heights))
print("peak_times: ",len(peak_times))
print("trigger_times: ",len(trigger_times))
print("areas: ",len(areas))

data_to_save = {
'trigger_time':trigger_times,
'pulse_height':pulse_heights,
'peak_times':peak_times,
'area':area,
} 


    np.savez(f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch{i}_{j}.npz',**data_to_save) 
     
    file.close """
    
""" 
  plt.figure()
    for k in range(20):
      plt.plot(time,pulse[k])
      plt.scatter(trigger_times[k],1.2,color = "tab:orange",label = "trigger time",alpha = 0.5)
      plt.scatter(peak_times[k],pulse_heights[k],color = "tab:green",label = "peak time",alpha = 0.5)
      plt.axhline(pulse_heights[k],color = "tab:purple",label ="pulse height value",alpha = 0.5)
    plt.xlabel("time(s)")
    plt.ylabel("voltage(V)")
    plt.legend()
    plt.show()  
     """

""" print(len(pulse_heights))
print(len(trigger_times))
print("pulse height",pulse_heights)
print("trigger_time",trigger_times)
 """


