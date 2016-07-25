from PIL import Image
import eye
import random
import numpy as np
import matplotlib.pyplot as plt

"""
KEY

item   - single image on final output figure
offset

"""
def polar2cart(r, theta):
    """
    Converts polar coordinates to cartesian coordinates. 
    Different frames of references in which the eye movements are described
    
    Parameters
    ----------
    r : array_like
        Radial coordinate *radius
    theta : array_like
        Angular coordinate *angle
    
    Returns
    ----------
    (x,y) : array_like
        Cartesian coordinates *position
    """
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return (x, y)


def get_positions(nitems, pos_offset, radius=400):
    """
    Determines [x,y] positions for search items given the number of items and 
    the radius of search circle in pixel
    
    Parameters
    -----------    
    nitems : int
        Number of items
    pos_offset : int
        Position of offset ****
    radius : int, optional
        Radius with default value
    
    Returns
    ----------
    screen positions : dict
        x & y coordinates for each item
    
    """
    
    positions     = np.arange(1, 360, 360/float(nitems))
    positions_off = positions + pos_offset
    
    screen_positions = {'x': [], 'y': []}
    
    for k in np.arange(nitems):
        #print k, positions_off[k]
        x, y = polar2cart(radius, positions_off[k])
        screen_positions['x'].append(x)
        screen_positions['y'].append(y)
    return screen_positions


def create_display(log_data, trl, target=False):
    """
    
    
    Parameters
    -----------
    log_data
    trl
    target
    
    Returns
    -----------
    im_base
    
    """
    conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}
    
    cond    = np.int(log_data['target'][log_data['trl'] == trl])
    n_items = log_data['nitems'][log_data['trl'] == trl]
    offset  = log_data['rot_offset'][log_data['trl'] == trl]
    positions = get_positions(n_items, offset)
    
    targ_x = log_data['tpos_x'][log_data['trl'] == trl]
    targ_y = log_data['tpos_y'][log_data['trl'] == trl]
    
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
bdata = eye.data_to_dict('%s/%s_%d' %(id, id, sess))
# Find which trials the target is correctly found (Where button response = Target presence)
bdata['correct'] = np.array(bdata['button']-6 == bdata['search'], dtype=int)
# Import eye tracking data [ block, time, x position, y position]
edata = np.loadtxt('%s/%s_%d_eye.txt' %(id, id, sess), skiprows=1)
# Extract corrected condition
bdata = eye.get_subset(bdata, 'correct', 1)


# loop through each condition
# for target in conditions.keys():

target = 0
sdata = eye.get_subset(bdata, 'target', target) 
trl_select = sdata['trl'][np.logical_and(sdata['search']==0, sdata['nitems']==n_items)]

f1 = plt.figure()

for k, trl in enumerate(trl_select):
    trl_disp = create_display(bdata, trl )
    eye_trial = edata[edata[:,0] == trl,]
    
    plt.subplot(4,5,k+1)
    plt.imshow(trl_disp, cmap = 'gray', vmin=0, vmax=255)
    plt.hold(True)
    plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
    plt.axis([300, 1660, 50, 1100])
    plt.suptitle(conditions[target])

f1.set_size_inches([24., 13.6375])

plt.savefig('figures/%s_%d_%s_%d.png' %(id, sess, conditions[target], n_items))

#f1 = plt.figure()
#trl_disp = create_display(bdata, trl )
#eye_trial = edata[edata[:,0] == trl,]
#plt.imshow(trl_disp, cmap = 'gray', vmin=0, vmax=255)
#plt.hold(True)
#plt.plot(eye_trial[:,2]+960, eye_trial[:,3]+600, 'b.')
#plt.axis([300, 1660, 50, 1100])


