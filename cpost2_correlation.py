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

""" 
plt.figure()
plt.hist(trigger_time_lf1,bins = 1000)
plt.show()

plt.figure()
plt.hist(trigger_time_lf3,bins = 1000)
plt.show() 
 """
""" 
plt.figure()
plt.hist(area1+area3,bins = 10000)
plt.show() 
"""

td = trigger_time_3 - 0.0009999983347107385

td_htrig  = trigger_time_1 - 0.0009999983347107385

td = trigger_time_3 - trigger_time_1
phd = pulse_height_3-pulse_height_1
td_lf = trigger_time_lf3 - trigger_time_lf1
#mask1 = (pulse_height_3<1.0)&(pulse_height_1<1.0)
mask2 = (-1.22 < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < -1.18)&(td*1e6>-10)&(td*1e6<10)
mask3 = (-1.40 < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < -1.30)&(td_lf*1e6>-15)&(td_lf*1e6<15)&(trigger_time_lf3>0.0009)&(trigger_time_lf1>0.0009)

""" 
plt.figure()
plt.scatter(trigger_time_lf1,trigger_time_lf3,s = 0.2)
plt.show() 
"""

plt.figure()
plt.title("mask: 1.3 ~ 1.4(V)")
plt.scatter(pulse_height_3[mask3],(td_lf)[mask3]*1e6,s = 0.2)
plt.xlabel("CH3 peak value(V)")
plt.ylabel("time diffrerence(μs)")
plt.savefig("cpost2_correlation_lf")
plt.show() 

""" 
plt.figure()
plt.hist(trigger_time_lf1*1e6,bins = 1000)
plt.semilogy()
plt.xlabel("CH1 trigger time(μs)")
plt.ylabel("count")
plt.xlim(0,4000)
plt.savefig("CH1_triggert_time_lf")
plt.show()

plt.figure()
plt.hist(trigger_time_lf3*1e6,bins = 1000)
plt.semilogy()
plt.xlabel("CH3 trigger time(μs)")
plt.ylabel("count")
plt.savefig("CH3_triggert_time_lf")
plt.show()
 """
""" fig, ax = plt.figure(), plt.axes()
cmap = plt.get_cmap('viridis')
scat = ax.scatter([], [], s=10)
def update(j):
    ax.clear()  # Clear the previous plot
    ax.set_title("final mask -1.22V to -1.18V")
    ax.set_xlabel(r'TES2 pulse height(V)')
    ax.set_ylabel(r'Trigger time(msec)')
    ax.set_xlim(-1.1,0.6)
    ax.set_ylim(9.98,10.0065)
    for i in range(100):
        mask = (area1+area3>-0.0246+0.0000025*j)&(area1+area3<-0.024-0.0000015*j)&(td*1e6>-10)&(td*1e6<10)
        colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
        ax.scatter(-pulse_height_3[mask], trigger_time_3[mask] * 1000, s=10, c=colors)
ani = FuncAnimation(fig, update, frames=range(100), interval=50)
ani.save('TES2_correlation_animation_areamask.gif', writer='pillow') 
 """


""" 
plt.figure()
plt.scatter(rise_time_1[mask2]*1e6,pulse_height_1[mask2],s = 0.2)
plt.xlim(0,50)
plt.xlabel("CH1 rise time(μs)")
plt.ylabel("CH1 peak value(V)")
#plt.savefig("risetime_ph1_mask")
plt.show()


plt.figure()
plt.scatter(rise_time_3[mask2]*1e6,pulse_height_3[mask2],s = 0.2)
plt.xlim(0,50)
plt.xlabel("CH3 rise time(μs)")
plt.ylabel("CH3 peak value(V)")
#plt.savefig("risetime_ph3_mask")
plt.show()
 """


""" 
fig, ax = plt.figure(), plt.axes()
ax.hist(pulse_height_3 + pulse_height_1, bins=1000)
ax.set_xlabel("total peak value(V)")
ax.set_ylabel("count")
ax.set_xlim(0.45, 2.3)
r = patches.Rectangle(xy=(1.16,0), width=0.08, height=1000, ec='tab:purple', fill=False , alpha = 0.8,linewidth = 1.5)
ax.add_patch(r)
r1 = patches.Rectangle(xy=(1.32,0), width=0.08, height=1000, ec='tab:orange', fill=False , alpha = 0.8,linewidth = 1.5)
ax.add_patch(r1) 
r2 = patches.Rectangle(xy=(1.53,0), width=0.03, height=1000, ec='tab:green', fill=False , alpha = 0.8,linewidth = 1.5)
ax.add_patch(r2) 
ax.set_yscale('log')
#fig.savefig("/Users/nozakirio/Desktop/analysis_23/cpost2_correlation_photo/total_peak_value_enlarge_mask_com.png")
plt.show() 
 """
  
""" 
fig, ax = plt.figure(), plt.axes()
cmap = plt.colormaps['viridis']
for i in range(1000):
    mask = mask_calc(pulse_height_1, pulse_height_3, -1.24, -1.16, 0.002, -1.0, i)
    colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
    ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=10, c=colors)
r = patches.Rectangle(xy=(0.6,-1.4), width=0.2, height=4, ec='tab:gray', fill=False , alpha = 0.7,linewidth  = 2)
ax.add_patch(r)
ax.set_title(f"total peak value mask:{1.40:.2f}V ~ {1.56:.2f}V , square range 0.6V ~ 0.8V")
ax.set_xlabel(r'CH3 pulse peak value(V)')
ax.set_ylabel(r'time difference (μsec)')
ax.set_ylim(-10,8)
ax.set_xlim(0,1.1)
ax.text(0.05, 0.95, 'green mask', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='tab:green',alpha = 0.7)
fig.savefig("CH3_correlation_mask4_1")
plt.show() 
 """

""" 
u = -1.16
l = -1.24
number = 0
def linear_func(x, a, b):
    return a * x + b
mask2 = (l < -pulse_height_1 - pulse_height_3) & (-pulse_height_1 - pulse_height_3 < u) & (td * 1e6 > -10) & (td * 1e6 < 10) & (0.4 < pulse_height_3) & (pulse_height_3 < 0.8)
popt, pcov = curve_fit(linear_func, pulse_height_3[mask2], td[mask2] * 1e6)
a_fit, b_fit = popt
fig, ax = plt.figure(), plt.axes()
cmap = plt.colormaps['viridis']
for i in range(1000):
    mask = mask_calc(pulse_height_1, pulse_height_3, l, u, 0.002, -1.0, i)
    colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
    ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=2, c=colors)
x_fit = np.linspace(0, 1.1, 500)  
y_fit = linear_func(x_fit, a_fit, b_fit)
ax.plot(x_fit, y_fit, color='tab:red', label='Fitted line')
fit_equation = f"y = {a_fit:.2e}x + {b_fit:.2e}"
ax.annotate(fit_equation, xy=(0.50, 0.1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
ax.set_title(f"total peak value mask: {-u:.2f}V ~ {-l:.2f}V, fitting part 0.4V ~ 0.8V")
ax.set_xlabel(r'CH3 pulse peak value(V)')
ax.set_ylabel(r'time difference (μsec)')
ax.set_ylim(-10, 8)
ax.set_xlim(0, 1.1)
ax.text(0.05, 0.95, 'purple mask', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='tab:purple',alpha = 0.7)
fig.savefig(f"/Users/nozakirio/Desktop/analysis_23/cpost2_correlation_photo/CH3/maskbytwo/CH3_correlation_mask{number}_fit")
plt.close(fig)

 """

""" u= -1.54
l = -1.56
number =100
def linear_func(x, a, b):#フィッティング関数
      return a * x + b
    # フィッティングの実行
mask2 = (l < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < u)&(td*1e6>-10)&(td*1e6<10)&(0.6<pulse_height_3)&(pulse_height_3<0.8)
popt, pcov = curve_fit(linear_func, pulse_height_3[mask2], td[mask2]*1e6)
a_fit, b_fit = popt
title = f" total peak value mask: {-u:.2f}V ~ {-l:.2f}V  , "
mask1 = (l < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < u)
fig, ax = plt.figure(), plt.axes()
cmap = plt.colormaps['viridis']
for i in range(1000):
    mask = mask_calc(pulse_height_1, pulse_height_3, l, u, 0.002, -1.0, i)
    colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
    ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=2, c=colors)
percentage = (len(pulse_height_3[mask1]) / len(pulse_height_3)) * 100
formatted_percentage = "{:.4f}".format(percentage)
ax.plot(pulse_height_3[mask2], linear_func(pulse_height_3[mask2], a_fit, b_fit), color='tab:red', label='Fitted line')
fit_equation = f"y = {a_fit:.2e}x + {b_fit:.2e}"
ax.annotate(fit_equation, xy=(0.50, 0.1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
    
ax.set_title(f"total peak value mask:{-u:.2f}V ~ {-l:.2f}V , fitting part 0.6V ~ 0.8V")
#ax.set_title(title + "ratio of data: " +formatted_percentage+"%")
ax.set_xlabel(r'CH3 pulse peak value(V)')
ax.set_ylabel(r'time difference (μsec)')
ax.set_ylim(-10,8)
ax.set_xlim(0,1.1)
fig.savefig(f"/Users/nozakirio/Desktop/analysis_23/cpost2_correlation_photo/CH3/mask_larger/CH3_correlation_addmask{number}")
plt.close(fig)
 """
""" 
for i in range(0,3,1):
    u= -1.16
    l = -1.24-0.16*i
    number =i
    title = f" total peak value mask: {-u:.2f}V ~ {-l:.2f}V  , "
    mask1 = (l < -pulse_height_1-pulse_height_3) & (-pulse_height_1-pulse_height_3 < u)
    fig, ax = plt.figure(), plt.axes()
    cmap = plt.colormaps['viridis']
    for i in range(1000):
        mask = mask_calc(pulse_height_1, pulse_height_3, l, u, 0.002, -1.0, i)
        colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
        ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=2, c=colors)
    percentage = (len(pulse_height_3[mask1]) / len(pulse_height_3)) * 100
    formatted_percentage = "{:.4f}".format(percentage)
    ax.set_title(title + "ratio of data: " +formatted_percentage+"%")
    ax.set_xlabel(r'CH3 pulse peak value(V)')
    ax.set_ylabel(r'time difference (μsec)')
    ax.set_ylim(-10,8)
    ax.set_xlim(0,1.1)
    fig.savefig(f"/Users/nozakirio/Desktop/analysis_23/cpost2_correlation_photo/CH3/mask_larger/CH3_correlation_addmask{number}")
    plt.close(fig)


    fig, ax = plt.figure(), plt.axes()
    cmap = plt.colormaps['viridis']
    for i in range(1000):
        mask = mask_calc(pulse_height_1, pulse_height_3, l, u, 0.002, -1.0, i)
        colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
        ax.scatter(pulse_height_3[mask], pulse_height_1[mask] , s=10, c=colors)
    percentage = (len(pulse_height_3[mask1]) / len(pulse_height_3)) * 100
    formatted_percentage = "{:.4f}".format(percentage)
    ax.set_title(title + "ratio of data: " +formatted_percentage+"%")
    ax.set_xlabel(r'CH3 pulse peak value(V)')
    ax.set_ylabel(r'CH1 pulse peak value(V)')
    ax.set_xlim(-0.1,1.05)
    ax.set_ylim(-0.1,1.3)
    fig.savefig(f"/Users/nozakirio/Desktop/analysis_23/cpost2_correlation_photo/CH3/mask_larger/ph3_ph1_correlation_addmask{number}")
    plt.close(fig)
 """


""" fig, ax = plt.figure(), plt.axes()
cmap = plt.colormaps['viridis']
for i in range(1000):
    mask = mask_calc(pulse_height_1, pulse_height_3, -1.22, -1.18, 0.002, -1.0, i)
    colors = cmap(pulse_height_3[mask] / np.max(pulse_height_3))  
    ax.scatter(pulse_height_3[mask], td[mask] * 1e6, s=10, c=colors)
ax.set_title("mask -1.V to -1.V")
ax.set_xlabel(r'CH3 pulse peak value(V)')
ax.set_ylabel(r'time difference (μsec)')
ax.set_ylim(-10,8)
fig.savefig("CH3_correlation_add1")
plt.show()
 """

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
