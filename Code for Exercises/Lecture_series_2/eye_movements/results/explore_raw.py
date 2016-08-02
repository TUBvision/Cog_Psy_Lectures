
import aux_funcs as a_eye
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
This script takes the results data and converts it into an appropriate format 
for plotting

"""




id = 'kvdb' # file label
sess = 4    # session number [from conditions]
n_items = 8 # number of items (figures) in stimuli

conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}

# Import computer and button response data as dictionary
bdata = a_eye.data_to_dict('%s/%s_%d' %(id, id, sess))
# Find which trials the target is correctly found (Where button response = Target presence)
bdata['correct'] = np.array(bdata['button']-6 == bdata['search'], dtype=int)
# Extract corrected condition
bdata = a_eye.get_subset(bdata, 'correct', 1)
# Import eye tracking data [ block, time, x position, y position]
edata=pd.read_csv('%s/%s_%d_eye.txt' %(id, id, sess),names=['blk','time','xpos','ypos'],skiprows=1,sep=' ')
edata=np.asarray(edata)

# loop through each condition, plotting each condition stimuli
for target in conditions.keys():
    sdata = a_eye.get_subset(bdata, 'target', target) 
    # Select trials to plot
    trial_select = sdata['trl'][np.logical_and(sdata['search']==0, sdata['nitems']==n_items)]
    
    # Initiate plotting
    f1 = plt.figure()
    for k, trial in enumerate(trial_select):
        trial_disp = create_display(bdata, trial )
        eye_trial = edata[edata[:,0] == trial,]
        
        plt.subplot(4,5,k+1)
        plt.imshow(trial_disp, cmap = 'gray', vmin=0, vmax=255)
        plt.hold(True)
        plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
        plt.axis([300, 1660, 50, 1100])
        plt.suptitle(conditions[target])
    
    f1.set_size_inches([24., 13.6375])
    
    #Option of saving figure (Takes up to 1 minute for all conditons)
    #plt.savefig('figures/%s_%d_%s_%d.png' %(id, sess, conditions[target], n_items))





#f1 = plt.figure()
#trial_disp = create_display(bdata, trial )
#eye_trial = edata[edata[:,0] == trial,]
#plt.imshow(trial_disp, cmap = 'gray', vmin=0, vmax=255)
#plt.hold(True)
#plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
#plt.axis([300, 1660, 50, 1100])


