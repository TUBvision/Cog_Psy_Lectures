import quant_funcs as qf
import aux_funcs as af
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This script contains plotting tools for velocity based, dispersion based and
moving average based thresholded fixation identification.
"""

id    = 'to'
sess  = 4
bdata = af.data_to_dict('%s/%s_%d' %(id, id, sess))
edata=pd.read_csv('%s/%s_%d_eye.txt' %(id, id, sess),names=['blk','time','xpos','ypos'],skiprows=1,sep=' ')
edata=np.asarray(edata)

trl = 1
# single trial data
eye_trial = edata[edata[:,0] == trl,]

# convert eye positions from pixels into deg visual angle
points = af.mm_to_visangle( af.pixel_to_mm( eye_trial[:,2:4] ) )


## plot velocity_based_identification
velocity_threshold=50
min_duration=100
smooth_window=100
sampling_rate=1000
fixations_veloc = qf.velocity_based_identification(points, velocity_threshold, min_duration, smooth_window, sampling_rate)
plt.figure(1)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations_veloc[:,3], fixations_veloc[:,4], 'ro')
plt.title('Velocity based')


## plot dispersion_based_identification
dispersion_threshold=100 #[100-200]ms
fixations_disp = qf.dispersion_based_identification(points, dispersion_threshold, min_duration, sampling_rate)
plt.figure(2)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations_disp[:,3], fixations_disp[:,4], 'ro')
plt.title("Dispersion based")


# plot moving_average_based_threshold
velocity = qf.calc_veloc(points * 1000)
velocity_smooth = qf.moving_average(velocity, sampling_rate, smooth_window, 'same')
plt.figure(3)
plt.plot(velocity_smooth)
plt.plot([0,1800], np.ones(2) * velocity_threshold)
velocity_smooth = qf.moving_average(velocity, sampling_rate, 40, 'same')
plt.plot(velocity_smooth)
plt.title("Moving average based")


n_items = 8
target = 0

sdata = af.get_subset(bdata, 'target', target)
trl_select = sdata['trl'][np.logical_and(sdata['search']==0, sdata['nitems']==n_items)]


plt.figure(4)
for count, trl in enumerate(trl_select):
    eye_trial = edata[edata[:,0]==trl,:]
    # convert eye positions from pixels into deg visual angle
    points = af.mm_to_visangle( af.pixel_to_mm( eye_trial[:,2:4] ) )
    
    plt.plot(points[:,0], points[:,1])
    plt.axis([-24, 24, -15, 15])

    plt.subplot(4,5,count+1)
    plt.plot(eye_trial[:,2], eye_trial[:,3])
    plt.xlim([-500,500])
    plt.ylim([-500,500])

