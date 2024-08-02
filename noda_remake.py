import os
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
import pytes.Util
import h5py

""" t1,p1=pytes.Util.fopen('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/b64/CH1_b64_190202p.fits') #11700波形存在
t3,p3=pytes.Util.fopen('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/b64/CH3_b64_190202p.fits')
 """ 

 
t1,p1=pytes.Util.fopen('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/b64/CH1_b64_190203p.fits')
t3,p3=pytes.Util.fopen('/Users/nozakirio/Desktop/CH1-63uA_CH3-73uA/b64/CH3_b64_190203p.fits')
 
 
 
def mask_calc(TES1_pulse,TES3_pulse,lower_limit,upper_limit,one_block, offset,i):
    mask = (offset + one_block *i <-TES3_pulse.min(axis=-1))&(-TES3_pulse.min(axis=-1)< offset + one_block *(i+1))&(lower_limit < (-TES1_pulse.min(axis=-1)-TES3_pulse.min(axis=-1))/2) & ((-TES1_pulse.min(axis=-1)-TES3_pulse.min(axis=-1))/2 < upper_limit)
    return mask

def moving_average_filter(pulse, window_size):
    window = np.ones(window_size) / float(window_size)
    smooth_pulse = np.convolve(pulse, window, mode = 'same')
    return smooth_pulse

def calc_trigger_time_ph(pulse,time,mask,arrival_time,window_size):
  if len(pulse[mask])!= 0:
    for j in range(pulse[mask].shape[0]):
      """立ち上がり前の電圧でトリガーを計算"""
      pre_event_point = np.where(time < arrival_time)
      mask_pulse = pulse[mask]
      pulse_average_before_event = np.average(mask_pulse[j][pre_event_point])
      pulse_trigger = -pulse_average_before_event + 0.05
      
      """平滑化"""
      smooth_pulse = moving_average_filter(-mask_pulse[j],window_size)

      """trigger_timeを計算"""
      trigger_point = np.where((smooth_pulse[1:] > pulse_trigger) & (smooth_pulse[0:-1] < pulse_trigger))[0]
      if len(trigger_point) != 0 : 
        trigger_time = time[trigger_point][0]
        pulse_height_p3 = np.min(mask_pulse[j])
        return trigger_time,pulse_height_p3
    else:
       pass

trigger_time = []
pulse_height = []
for i in range(1000):
    mask = mask_calc(p1,p3,-0.58,-0.555,0.0018,-1.0,i)
    result = calc_trigger_time_ph(p3, t3, mask, 0.00002, 25)
    if result is not None:
      trigger_time.append(result[0]) 
      pulse_height.append(result[1])
        
trigger_time = np.array(trigger_time)
pulse_height = np.array(pulse_height)

plt.figure()
plt.scatter(pulse_height,trigger_time*1000,s = 2)
plt.xlabel("CH3 pulse height(V)")
plt.ylabel("trigger time(msec)")
plt.savefig("noda_remake_203")
  