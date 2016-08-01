import matplotlib.pyplot as plt
import aux_funcs as af
import quant_funcs as qf
import pandas as pd
import numpy as np
"""
--- KEY ---

search item - single figure on output stimuli
offset - rotational offset of search items 
"""

"""
Import raw data
"""
id    = 'to' # initials for trial participant
sess  = 4    # session number

# Import computer data as dictionary containing stimulis given and button responses
bdata = af.data_to_dict('%s/%s_%d' %(id, id, sess)) 
"""
Typing "bdata.keys()" gives you:
['rt',
 'search',
 'nitems',
 'trl',
 'button',
 'tpos_x',
 'rot_offset',
 'blk',
 'tpos_y',
 'target']
 
rt - reaction time
search - absence/presence [0,1]
nitems - number of figures in stimuli
trl - trial number
button - absent/present participant response [6,7]
rot_offset - offset angles for presence of odd-one-out
blk -  block number
tpos_y - 
target - [0,1,2,3] (Rotated outwards, illusionary, filled gray, square)

"""
# Import eye tracking data [ block, time , x position , y position]
edata=pd.read_csv('%s/%s_%d_eye.txt' %(id, id, sess),names=['blk','time','xpos','ypos'],skiprows=1,sep=' ')
edata=np.asarray(edata)

trl = 1 # single trial data
eye_trial = edata[edata[:,0] == trl,] # extract trial number
points = af.mm_to_visangle( af.pixel_to_mm( eye_trial[:,2:4] ) ) # convert pixels to degrees of visual angle

# 4 conditions of different target types 
conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}

"""
Question 1
"""
# Edit these Variables
nitems = [4,8,12] # number of figure items
search = 0 # absence/presence [0,1]
target = 0 # target coniditon type

# get subset data from bdata
sdata = af.get_subset(bdata, 'target', target) 

plt.figure(1)

# Select trials to plot
trial_select=np.zeros((20,3))
for n in range(3):

    trial_select[:,n] = sdata['trl'][np.logical_and(sdata['search']==search, sdata['nitems']==nitems[n])]
    
    # Initiate plotting    
    for trl in range(3):
        trial = trial_select[trl,n]
        print trial
        trial_disp = af.create_display(bdata, trial )
        eye_trial = edata[edata[:,0] == trial,]
        
        plt.subplot(3,3,1)
        plt.imshow(trial_disp, cmap = 'gray', vmin=0, vmax=255)
        plt.hold(True)
        plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
        plt.axis([300, 1660, 50, 1100])
        plt.suptitle(conditions[target])




"""
Currently an Error in the subplotting routine. Save trial disp and plot afterwards perhaps.

"""








#
#
#
#
## Plot eye movement path
#plt.figure(1)
#plt.plot(points[:,0],points[:,1])
#plt.title("eye movement path")
#
## Plot velocities of path
#plt.figure(2)
#plt.plot(qf.calc_veloc(points),label='raw velocities')
#
## smooth raw velocities with smoothing window
#smooth_window=100
#sampling_rate=1000
#velocity = qf.moving_average(qf.calc_veloc(points), sampling_rate, smooth_window, 'same')
#plt.plot(velocity, label='smoothed velocities')
#
## isolate velocities less than velocity threshold as fixations
#velocity_threshold = 0.1 # 0.1 == 100ms
#fix_velocities = np.array(velocity < velocity_threshold, dtype = int)
#plt.plot(fix_velocities, label='thresholded velocities')
#
## difference between subsequent velocities (out[n] = a[n+1] - a[n])
#fixation_index = np.diff(fix_velocities)
##plt.plot(fixation_index,label='difference velocities')
#
## locations of where a change in velocity == 1
## np.where()[0] gets array of locations
#fix_start = np.where(fixation_index == 1)[0]+1
## assumption that eye movements start with fixation (i.e. velocity zero)
#fix_start = np.r_[0, fix_start]
#
#
## add 1 index because difference-vector index shifted by one element
#nsamples = len(velocity)    
#fix_end   = np.nonzero(fixation_index == -1)[0]+1
#fix_end   = np.r_[fix_end, nsamples]
#
#if fix_velocities[-1] == 0:
#    fix_end = fix_end[:-1]
#
#if fix_velocities[0] == 0:
#    fix_start = fix_start[1:]
#
#fix_index = np.c_[fix_start, fix_end]
#
#
## eliminate fixations with DUR < dur_crit m
#duration_threshold=0.1
#critical_duration = duration_threshold * sampling_rate/1000
#fix_dur   = fix_index[:,1]-fix_index[:,0]
#fix_index = fix_index[fix_dur>critical_duration,:] # Indicies of fixation
##plt.plot(fix_index,label='next')
#plt.legend()
#
#fixations = af.fix_index_to_fixations(points, fix_index, sampling_rate)
#plt.plot(fixations)