from PIL import Image
import aux_funcs as a_eye
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
This script takes the results data and converts it into an appropriate format 
for plotting

--- KEY ---

search item - single figure on output stimuli
offset - rotational offset of search items 
"""

def get_positions(nitems, offset, radius=400):
    """
    Determines the [x,y] positions of search items given the number of items, 
    any rotational offset and the radius of search circle area.
    
    Parameters
    -----------    
    nitems : int
        Number of items
    offset : int
        Rotational offset of items
    radius : int, optional
        Radius with default value
    
    Returns
    ----------
    screen positions : dict
        x & y coordinates for each item
    
    """
    positions     = np.arange(1, 360, 360/float(nitems))
    positions_off = positions + offset
    
    screen_positions = {'x': [], 'y': []}
    
    for k in np.arange(nitems):
        #print k, positions_off[k]
        x, y = a_eye.polar2cart(radius, positions_off[k])
        screen_positions['x'].append(x)
        screen_positions['y'].append(y)
    return screen_positions


def create_display(log_data, trial, target=False):
    """
    Plots the background image stimuli, given the condition number, number of 
    items, item figure type and rotational offset, upon which the eye tracking 
    path is plotted.
    
    Parameters
    -----------
    log_data : array_like
        input data
    trial : int
        experimental trial number
    target : boolean, optional
        default - False
    
    Returns
    -----------
    im_base : array_like
        output stimuli (underlay)
    
    """
    conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}
    
    # "[log_data['trial'] == trial])" selects trial number
    cond    = np.int(log_data['target'][log_data['trl'] == trial])
    n_items = log_data['nitems'][log_data['trl'] == trial]
    offset  = log_data['rot_offset'][log_data['trl'] == trial]
    positions = get_positions(n_items, offset)
    
    targ_x = log_data['tpos_x'][log_data['trl'] == trial]
    targ_y = log_data['tpos_y'][log_data['trl'] == trial]
    
    im_base = Image.new('L', (1920, 1200), 128)
    if random.randint(0,1) == 1:
        dist_type = 'thin'
        #targ_type = 'fat'
    else:
        dist_type = 'fat'
        #targ_type = 'thin'
    
    im_distractor = Image.open('../stimuli/norm_%s_%s_10.bmp' %(dist_type, conditions[cond]))
    new_dist   = im_distractor.resize((110,110))
    
    loc_offset = 55
    for k in np.arange(np.int(n_items)):
        xpos = np.round(positions['x'][k])+960
        ypos = np.round(positions['y'][k])+600
        ul_x = np.int(min(xpos+loc_offset, xpos-loc_offset))
        ul_y = np.int(min(ypos+loc_offset, ypos-loc_offset))
        lr_x = np.int(max(xpos+loc_offset, xpos-loc_offset))
        lr_y = np.int(max(ypos+loc_offset, ypos-loc_offset))
        
        im_base.paste(new_dist, (ul_x, ul_y, lr_x, lr_y))
    
    if target:
        #im_target     = Image.open('../stimuli/norm_fat_aligned_10.bmp')
        #new_target = im_target.resize((110,110))
        t_ul_x = np.int(min(targ_x+55, targ_x-55))
        t_ul_y = np.int(min(targ_y+55, targ_y-55))
        t_lr_x = np.int(min(targ_x+55, targ_x-55))
        t_lr_y = np.int(min(targ_y+55, targ_y-55))
        im_base.paste(new_dist, (t_ul_x, t_ul_y, t_lr_x, t_lr_y))
    
    return im_base


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


