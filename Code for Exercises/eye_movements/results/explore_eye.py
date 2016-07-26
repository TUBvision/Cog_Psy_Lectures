import qunatification_eye as q_eye
import auxilliary_eye as a_eye
import numpy as np
import matplotlib.pyplot as plt

id    = 'to'
sess  = 4
bdata = a_eye.data_to_dict('%s/%s_%d' %(id, id, sess))
edata = np.loadtxt('%s/%s_%d_eye.txt' %(id, id, sess), skiprows=1)


trl = 1
# single trial data
eye_trial = edata[edata[:,0] == trl,]

# convert eye positions from pixels into deg visual angle
points = a_eye.mm_to_visangle( a_eye.pixel_to_mm( eye_trial[:,2:4] ) )

## plot velocity_based_identification
velocity_threshold=50
min_duration=100
smooth_window=100
sampling_rate=1000
fixations_veloc = q_eye.velocity_based_identification(points, velocity_threshold, min_duration, smooth_window, sampling_rate)

plt.figure(1)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations_veloc[:,3], fixations_veloc[:,4], 'ro')
plt.title('Velocity based')

## plot dispersion_based_identification
dispersion_threshold=100 #[100-200]ms
fixations_disp = q_eye.dispersion_based_identification(points, dispersion_threshold, min_duration, sampling_rate)

plt.figure(2)
plt.plot(points[:,0], points[:,1])
plt.plot(fixations_disp[:,3], fixations_disp[:,4], 'ro')
plt.title("Dispersion based")

# plot moving_average_based_threshold
velocity = q_eye.calc_veloc(points * 1000)
velocity_smooth = q_eye.moving_average(velocity, sampling_rate, smooth_window, 'same')

plt.figure(3)
plt.plot(velocity_smooth)
plt.plot([0,1800], np.ones(2) * velocity_threshold)
velocity_smooth = q_eye.moving_average(velocity, sampling_rate, 40, 'same')
plt.plot(velocity_smooth)
plt.title("Moving average based")


n_items = 8
target = 0

sdata = a_eye.get_subset(bdata, 'target', target)
trl_select = sdata['trl'][np.logical_and(sdata['search']==0, sdata['nitems']==n_items)]


plt.figure(4)
for count, trl in enumerate(trl_select):
    eye_trial = edata[edata[:,0]==trl,:]
    # convert eye positions from pixels into deg visual angle
    points = a_eye.mm_to_visangle( a_eye.pixel_to_mm( eye_trial[:,2:4] ) )
    
    plt.plot(points[:,0], points[:,1])
    plt.axis([-24, 24, -15, 15])

    plt.subplot(4,5,count+1)
    plt.plot(eye_trial[:,2], eye_trial[:,3])
    plt.xlim([-500,500])
    plt.ylim([-500,500])

