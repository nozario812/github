import h5py
import os
import sys
import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as cr
from scipy.optimize import curve_fit
import optparse
import matplotlib.pylab as plab
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def mask_calc(TES1_peak_value,TES3_peak_value,lower_limit,upper_limit,one_block, offset,i):
    mask = (offset + one_block *i < -TES3_peak_value)&(-TES3_peak_value< offset + one_block *(i+1))&(lower_limit < -TES1_peak_value-TES3_peak_value) & (-TES1_peak_value-TES3_peak_value < upper_limit)
    return mask

hardware_triggertime = 0.010999981681818127

trigger_time_1 = []
pulse_height_1 = []
peak_time_1 = []
area1 = []
for i in range(1,11,1):
  file = f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch1_{i}.npz'
  loaded_data = np.load(file)
  trigger_time_1.append(loaded_data["trigger_time"])
  pulse_height_1.append(loaded_data["pulse_height"])
  peak_time_1.append(loaded_data["peak_times"])
  area1.append(loaded_data["area"])
trigger_time_1 = np.concatenate(trigger_time_1)
pulse_height_1 = np.concatenate(pulse_height_1)
peak_time_1 = np.concatenate(peak_time_1)
area1 = np.concatenate(area1)


trigger_time_3 = []
pulse_height_3 = []
peak_time_3 = []
area3 = []
for i in range(1,11,1):
  file = f'/Users/nozakirio/Desktop/analysis_23/trigger_time_pulse_height_npz/cpost2_ch3_{i}.npz'
  loaded_data = np.load(file)
  trigger_time_3.append(loaded_data["trigger_time"])
  pulse_height_3.append(loaded_data["pulse_height"])
  peak_time_3.append(loaded_data["peak_times"])
  area3.append(loaded_data["area"])
trigger_time_3 = np.concatenate(trigger_time_3)
pulse_height_3 = np.concatenate(pulse_height_3)
peak_time_3 = np.concatenate(peak_time_3)
area3 = np.concatenate(area3)

rise_time_1 = peak_time_1-trigger_time_1
rise_time_3 = peak_time_3-trigger_time_3

trigger_time_lf1 = []
trigger_time_lf3 = []
for i in range(1,11,1):
  file = f'/Users/nozakirio/Desktop/analysis_23/lf_trigtime_npz/cpost2_lf_trigtime_ch3_{i}.npz'
  loaded_data = np.load(file)
  trigger_time_lf3.append(loaded_data["trigger_time"])
  file = f'/Users/nozakirio/Desktop/analysis_23/lf_trigtime_npz/cpost2_lf_trigtime_ch1_{i}.npz'
  loaded_data = np.load(file)
  trigger_time_lf1.append(loaded_data["trigger_time"])
trigger_time_lf1 = np.concatenate(trigger_time_lf1)
trigger_time_lf3 = np.concatenate(trigger_time_lf3)


td = trigger_time_3 - 0.0009999983347107385

td_htrig  = trigger_time_1 - 0.0009999983347107385

td = trigger_time_3 - trigger_time_1
phd = pulse_height_3-pulse_height_1
td_lf = trigger_time_lf3 - trigger_time_lf1
mask1 = (pulse_height_3<1.0)&(pulse_height_1<1.0)
mask2 = (-1.22 < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < -1.18)&(td*1e6>-10)&(td*1e6<10)
mask3 = (-1.40 < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < -1.30)&(td_lf*1e6>-15)&(td_lf*1e6<15)&(trigger_time_lf3>0.0009)&(trigger_time_lf1>0.0009)



fig, ax = plt.figure(), plt.axes()
cmap = plt.colormaps['viridis']
for i in range(1000):
    mask = mask_calc(pulse_height_1, pulse_height_3, -1.22, -1.18, 0.002, -1.0, i)
    colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
    ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=10, c=colors)
ax.set_title("mask 1.18.V to 1.22V")
ax.set_xlabel(r'CH3 pulse peak value(V)')
ax.set_ylabel(r'time difference (μsec)')
ax.set_ylim(-10,8)
fig.savefig("CH3_correlation")
plt.show()


"""fig, ax = plt.figure(), plt.axes()
cmap = plt.get_cmap('viridis')
scat = ax.scatter([], [], s=10)
def update(j):
    ax.clear()  # Clear the previous plot
    ax.set_title("final mask -1.22V to -1.18V")
    ax.set_xlabel(r'TES2 pulse height(V)')
    ax.set_ylabel(r'Trigger time(msec)')
    ax.set_xlim(-1.1,0.6)
    ax.set_ylim(9.98,10.0065)
    for i in range(1000):
        mask = mask_calc(pulse_height_1, pulse_height_3, -2.25+0.0103*j, -0.52-0.0066*j, 0.002, -1.0, i)
        colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
        ax.scatter(-pulse_height_3[mask], trigger_time_3[mask] * 1000, s=10, c=colors)
ani = FuncAnimation(fig, update, frames=range(100), interval=50)
ani.save('TES2_correlation_animation.gif', writer='pillow') 
 """
