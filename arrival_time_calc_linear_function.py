import numpy as np
from numba import jit

def find_index(pulse, pulse_height, threshold_rate):
    return np.array([
        np.min(np.argwhere(p > ph * threshold_rate)) if np.argwhere(p > ph * threshold_rate).size > 0 else 0
        for p, ph in zip(pulse, pulse_height)
    ])

def calculate_tr_voltage(pulse, pulse_height, delta_time, threshold_rate1, threshold_rate2):
    upper_index = find_index(pulse, pulse_height, threshold_rate1)
    upper_time = upper_index * delta_time
    upper_voltage = np.array([p[idx] for p, idx in zip(pulse, upper_index)])
    
    below_index = find_index(pulse, pulse_height, threshold_rate2)
    below_time = below_index * delta_time
    below_voltage = np.array([p[idx] for p, idx in zip(pulse, below_index)])
    
    return upper_time, upper_voltage, below_time, below_voltage


def calculate_tr_time(pretrigger, upper_voltage,  upper_time,below_voltage, below_time):
    if np.array_equal(upper_time, below_time) or np.array_equal(below_voltage,upper_voltage) or np.any(upper_time) == 0:
        arrival_time = below_time
    elif np.any(below_time) == 0:
        arrival_time = upper_time
    else:
        numerator = pretrigger - ((upper_voltage * below_time - below_voltage * upper_time) / (below_time - upper_time))
        denominator = (below_voltage - upper_voltage) / (below_time - upper_time)
        arrival_time = numerator / denominator 
    return arrival_time

#github動作確認#