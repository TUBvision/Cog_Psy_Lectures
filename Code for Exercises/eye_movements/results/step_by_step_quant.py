# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:02:52 2016

@author: will
"""
import matplotlib.pyplot as plt
#import aux_funcs as af
#import quant_funcs as qf
import pandas as pd
import numpy as np

# Import raw data
id    = 'to'
sess  = 4
bdata = af.data_to_dict('%s/%s_%d' %(id, id, sess))
edata=pd.read_csv('%s/%s_%d_eye.txt' %(id, id, sess),names=['blk','time','xpos','ypos'],skiprows=1,sep=' ')
edata=np.asarray(edata)
trl = 1 # single trial data
eye_trial = edata[edata[:,0] == trl,]
points = af.mm_to_visangle( af.pixel_to_mm( eye_trial[:,2:4] ) )


# Plot eye movement path
plt.figure(1)
plt.plot(points[:,0],points[:,1])
plt.title("eye movement path")

# Plot velocities of path
plt.figure(2)
plt.plot(qf.calc_veloc(points),label='raw velocities')

# smooth raw velocities with smoothing window
smooth_window=100
sampling_rate=1000
velocity = qf.moving_average(qf.calc_veloc(points), sampling_rate, smooth_window, 'same')
plt.plot(velocity, label='smoothed velocities')

# isolate velocities less than velocity threshold as fixations
velocity_threshold = 0.1 # 0.1 == 100ms
fix_velocities = np.array(velocity < velocity_threshold, dtype = int)
plt.plot(fix_velocities, label='thresholded velocities')

# difference between subsequent velocities (out[n] = a[n+1] - a[n])
fixation_index = np.diff(fix_velocities)
#plt.plot(fixation_index,label='difference velocities')

# locations of where a change in velocity == 1
# np.where()[0] gets array of locations
fix_start = np.where(fixation_index == 1)[0]+1
# assumption that eye movements start with fixation (i.e. velocity zero)
fix_start = np.r_[0, fix_start]


# add 1 index because difference-vector index shifted by one element
nsamples = len(velocity)    
fix_end   = np.nonzero(fixation_index == -1)[0]+1
fix_end   = np.r_[fix_end, nsamples]

if fix_velocities[-1] == 0:
    fix_end = fix_end[:-1]

if fix_velocities[0] == 0:
    fix_start = fix_start[1:]

fix_index = np.c_[fix_start, fix_end]


# eliminate fixations with DUR < dur_crit m
duration_threshold=0.1
critical_duration = duration_threshold * sampling_rate/1000
fix_dur   = fix_index[:,1]-fix_index[:,0]
fix_index = fix_index[fix_dur>critical_duration,:] # Indicies of fixation
#plt.plot(fix_index,label='next')
plt.legend()

fixations = af.fix_index_to_fixations(points, fix_index, sampling_rate)
plt.plot(fixations)