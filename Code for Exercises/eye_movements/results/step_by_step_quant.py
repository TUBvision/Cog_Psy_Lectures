import matplotlib.pyplot as plt
import aux_funcs as af
import quant_funcs as qf
import pandas as pd
import numpy as np


"""
Question 1 - Understanding the raw data
"""
id    = 'to' # initials for trial participant
sess  = 4    # session number

# Import computer data as dictionary containing stimulis given and button responses
bdata = af.data_to_dict('%s/%s_%d' %(id, id, sess)) 

"""
Typing "bdata.keys()" gives you:
['rt', 'search', 'nitems', 'trl', 'button', 'tpos_x', 'rot_offset', 'blk', 'tpos_y', 'target']
 
rt         - reaction time
search     - absence/presence [0,1]
nitems     - number of figures in stimulus
trl        - trial number
button     - absent/present participant response [6,7]
rot_offset - offset angles for presence of odd-one-out
blk        - block number
tpos_y     - ?????
target     - [0,1,2,3] (Rotated outwards, illusionary, filled gray, square)
"""

# Import eye tracking data [ block, time , x position , y position]
edata=pd.read_csv('%s/%s_%d_eye.txt' %(id, id, sess),names=['blk','time','xpos','ypos'],skiprows=1,sep=' ')
edata=np.asarray(edata)

trl = 1 # single trial data
eye_trial = edata[edata[:,0] == trl,] # extract trial number
points = af.mm_to_visangle( af.pixel_to_mm( eye_trial[:,2:4] ) ) # convert pixels to degrees of visual angle

# 4 conditions of different target types 
conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}

# Variables
nitems = [4,8,12] # number of figure items
search = 1 # absence/presence [0,1]
target = 0 # target coniditon type

# get subset data from bdata
sdata = af.get_subset(bdata, 'target', target) 

plt.figure(1)
step = 1 # plotting step
trial_select=np.zeros((20,3)) # extract 20 trials for 3 different nitems

for n in range(3): # loop through number of items
    
    # select trial based on presence of target and number of items
    trial_select[:,n] = sdata['trl'][np.logical_and(sdata['search']==search, sdata['nitems']==nitems[n])]
    
    # Plotting routine  
    for trl in range(3):
        trial = trial_select[trl,n]
        trial_disp = af.create_display(bdata, trial )
        eye_trial = edata[edata[:,0] == trial,]
        
        plt.subplot(3,3,step)
        step=step+1
        plt.imshow(trial_disp, cmap = 'gray', vmin=0, vmax=255)
        plt.hold(True)
        plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
        plt.axis([300, 1660, 50, 1100])
        plt.suptitle(conditions[target])
        

"""
Question 2 - Path quantification - velocity based

Note: the below is an expanded version of qf.velocity_based_identification()
"""
# Variables
smooth_window=100
sampling_rate=1000
velocity_threshold = 0.1 # 0.1 == 100ms
duration_threshold=0.1 


# Plot eye movement path
plt.figure(2)
plt.subplot(1,3,1)
plt.plot(points[:,0],points[:,1])
plt.title("path")
plt.tick_params(axis='both',which='both', bottom='off',top='off', left='off',right='off', labelbottom='off',labelleft='off')

# Plot velocities of path via calc_velc function.
plt.subplot(1,3,2)
plt.title("processing")
plt.plot(qf.calc_veloc(points),label='raw velocities')

# Smooth raw velocities with smoothing window (cleaning up the data)
velocity = qf.moving_average(qf.calc_veloc(points), sampling_rate, smooth_window, 'same')
plt.plot(velocity, label='smoothed velocities')

# Isolate velocities less than velocity threshold as fixations
fix_velocities = np.array(velocity < velocity_threshold, dtype = int)
plt.plot(fix_velocities, label='thresholded velocities')
plt.legend()


"""
??????????????????
"""
# difference between subsequent fixations (out[n] = a[n+1] - a[n])
fixation_index = np.diff(fix_velocities)
# locations of where a change in velocity == 1 (np.where()[0] gets array of locations)
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
critical_duration = duration_threshold * sampling_rate/1000
fix_dur   = fix_index[:,1]-fix_index[:,0]
fix_index = fix_index[fix_dur>critical_duration,:] # Indicies of fixation

# Convert indices into positions
fixations = af.fix_index_to_fixations(points, fix_index, sampling_rate)
plt.subplot(1,3,3)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations[:,3], fixations[:,4], 'ro')
plt.title('path with fixations')


"""
Question 3 - Path quantification - dispersion based 

Note: As fixations are low velocity, they tend to cluster. This method classifies
fixations based on maximum cluster seperation. In other words, it computes
fixations based on dispersion of sample points and minimum duration
"""
# Variables
min_duration=100
dispersion_threshold=1 #[100-200]ms

# Extract fixations used dispersion based method
fixations_disp = qf.dispersion_based_identification(points, dispersion_threshold, min_duration, sampling_rate)

plt.figure(3)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations_disp[:,3], fixations_disp[:,4], 'ro')
plt.title("Dispersion based")
plt.tick_params(axis='both',which='both', bottom='off',top='off', left='off',right='off', labelbottom='off',labelleft='off')


