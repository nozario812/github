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


