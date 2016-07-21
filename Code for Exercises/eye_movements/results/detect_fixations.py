import numpy as np
import sys
import eye
import time
import matplotlib.pyplot as plt
import numpy as np

# parameter
sampling_rate = 1000.0

# minimum fixation duration in ms
min_duration  = 100

# smoothing window in ms
smooth_window = 30

# velocity threshold indeg/sec
velocity_threshold = 50

# dispersion threshold in deg visual angle
dispersion_threshold = 1


## save eye movement data for all trials
id = sys.argv[1]
fix_algorithm = sys.argv[2]

sess = 4
bdata = eye.data_to_dict('%s/%s_%d' %(id, id, sess))
edata = np.loadtxt('%s/%s_%d_eye.txt' %(id, id, sess), skiprows=1)

fix_out = np.array([])

for trl in np.arange(1, 481):
    print('working on %s fixation_type %s' %(id, fix_algorithm))
    eye_trial = edata[edata[:,0] == trl,]
    # convert eye positions from pixels into deg visual angle
    points = eye.mm_to_visangle( eye.pixel_to_mm( eye_trial[:,2:4] ) )
    
    ## blink detection and exclusion
    n_samples = points.shape[0]
    points_clean = eye.del_artefact(points, xlim=24, ylim=15)
    n_samples_clean = points_clean.shape[0]
    if (n_samples - n_samples_clean) > 0:
        fixations = eye.nans((1,5))
    else:
        if fix_algorithm == 'velocity':
            fixations = eye.velocity_based_identification(points, velocity_threshold, min_duration,  smooth_window, sampling_rate)
        elif fix_algorithm == 'dispersion':
            fixations = eye.dispersion_based_identification(points, dispersion_threshold, min_duration,  sampling_rate)
    
    if fix_out.shape[0] == 0:
        fix_out = np.c_[np.ones(fixations.shape[0]) * trl, fixations]
    else:
        fix_out = np.r_[fix_out, np.c_[np.repeat(trl, fixations.shape[0]), fixations]]


### save file
out = open('%s/%s_%d_%s.txt' %(id, id, sess, fix_algorithm), "w")
out.write('# minimum duration: %d\n'     %min_duration)
out.write('# smoothing window: %d\n'     %smooth_window)
out.write('# velocity threshold: %d\n'   %velocity_threshold)
out.write('# dispersion threshold: %d\n' %dispersion_threshold)
out.write('trl fix_id fdur sdur xpos ypos\n')

np.savetxt(out, fix_out, fmt='%d %1.2f %1.2f %1.2f %1.2f %1.2f', delimiter='\t')
out.flush()
out.close()



### plot data from one condition
#conditions = {0: 'mis', 1: 'al', 2: 'filled', 3: 'no'}

#sess    = 4
#n_items = 8

#for fix_algorithm in ['velocity', 'dispersion']:
    
    #for id in ['ipa', 'ipr', 'kvdb', 'sk', 'to', 'tp', 'vf']:
        #print id
        #bdata = eye.data_to_dict('%s/%s_%d' %(id, id, sess))
        #edata = np.loadtxt('%s/%s_%d_eye.txt' %(id, id, sess), skiprows=1)
        
        #for target in np.arange(4):
            #print target
            #sdata = eye.get_subset(bdata, 'target', target)
            #trl_select = sdata['trl'][np.logical_and(sdata['search']==0, sdata['nitems']==n_items)]
            
            #f1 = plt.figure()
            #for count, trl in enumerate(trl_select):
                #print trl
                #eye_trial = edata[edata[:,0] == trl,]
                ## convert eye positions from pixels into deg visual angle
                #points = eye.mm_to_visangle( eye.pixel_to_mm( eye_trial[:,2:4] ) )
                
                ### blink detection and exclusion
                #n_samples = points.shape[0]
                #points_clean = eye.del_artefact(points, xlim=24, ylim=15)
                #n_samples_clean = points_clean.shape[0]
                #if (n_samples - n_samples_clean) > 0:
                    #fixations = np.array([])
                #else:
                    #if fix_algorithm == 'velocity':
                        #fixations = eye.velocity_based_identification(points, velocity_threshold, min_duration,  smooth_window, sampling_rate)
                    #elif fix_algorithm == 'dispersion':
                        #fixations = eye.dispersion_based_identification(points, dispersion_threshold, min_duration,  sampling_rate)
                
                #plt.subplot(4,5,count+1)
                #plt.plot(points[:,0], points[:,1])
                #plt.title(trl)
                #if fixations.shape[0] != 0:
                    #plt.plot(fixations[:,3], fixations[:,4], 'ro')
                #plt.axis([-15,15,-15,15])
            
            #f1.set_size_inches([16,12])
            #plt.savefig('figures/samples_fix_%s_%s_%s_%d_%d.png' %(fix_algorithm, id, conditions[target], n_items, sess))
            #plt.close(f1)
