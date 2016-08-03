#!/usr/bin/env python
#coding: CP1250

## This experiments' purpose is to measure search time as a function of number of items and conditions
## shown are aligned - filled or unfilled - and misaligned Varin figures, search is for the odd one out
## 


import vsg_client
import numpy as np
import random
import time
import re, os, sys


def cart_to_polar(y, x):
    """
    converts cartesian to polar coordinates
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    return (r, theta)


def polar_to_cart(r, theta):
    """
    converts polar to cartesian coordinates
    """
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return (x, y)


def get_positions(nitems, radius):
    """
    determines [x,y]-positions for search items given the number of items and the radius of search circle in pixel
    """
    positions  = np.arange(1, 360, 360/float(nitems))
    pos_offset = random.randint(0, np.round(360/float(nitems))-1)
    positions_off = positions + pos_offset
    
    screen_positions = {'x': [], 'y': []}
    
    for k in np.arange(nitems):
        print k, positions_off[k]
        x, y = polar_to_cart(radius, positions_off[k])
        screen_positions['x'].append(x)
        screen_positions['y'].append(y)
    return screen_positions, pos_offset


def show_display(vsg, page_number, background, target, target_id, distractor_id, positions):
    """
    create search array for a single trial
    input
    =====
    background  - integer number specifying LUT entry
    target      - boolean whether target is present or not
    target_id   - identifier specifying shape of target: thin vs. fat
    distractor_id - identifier specifying shape of distractor: thin vs. fat
    positions   - dictionary containing list of x and y values specifying positions of search items
    """
    vsg.vsgSetDrawPage(page_number, background)
    nitems = len(positions['x'])
    target_pos = random.randint(0, nitems-1)
    targ_x, targ_y = 0,0
    for k in np.arange(nitems):
        x, y = positions['x'][k], positions['y'][k]
        if target_pos == k and target == 1:
            targ_x, targ_y = x, y
            vsg.vsgDrawImage('stimuli/%s.bmp' %target_id, (targ_x, targ_y))
        else:
            vsg.vsgDrawImage('stimuli/%s.bmp' %distractor_id, (x, y))
    
    vsg.vsgSetDisplayPage(page_number)
    vsg.cbboxFlush()
    return (targ_x, targ_y)


def show_display_intro(vsg, page_number, background, target_id, distractor_id, positions):
    """
    """
    vsg.vsgSetDrawPage(page_number, background)
    vsg.vsgDrawString(-290, 340, 'Finden Sie den Abweicher!')
    vsg.vsgDrawString(-120, 370, 'Alle gleich -> LINKS, Einer anders -> RECHTS')
    vsg.vsgDrawString(0, 0, '+')

    nitems = len(positions['x'])
    target_pos = 4
    for k in np.arange(nitems):
        x, y = positions['x'][k], positions['y'][k]
        if target_pos == k:
            targ_x, targ_y = x, y
            vsg.vsgDrawImage('stimuli/%s.bmp' %target_id, (targ_x, targ_y))
        else:
            vsg.vsgDrawImage('stimuli/%s.bmp' %distractor_id, (x, y))
    
    vsg.vsgSetDisplayPage(page_number)
    vsg.cbboxFlush()
    return


def show_fixation(vsg, page_number, background, positions):
    """
    """
    vsg.vsgSetDrawPage(page_number, background)
    vsg.vsgDrawString(positions[0], positions[1], '+')
    vsg.vsgSetDisplayPage(page_number)
    vsg.cbboxFlush()


def show_pause(vsg, page_number, background, progress):
    """
    display pause screen and show progress
    """
    vsg.vsgSetDrawPage(page_number, background)
    vsg.vsgDrawString(0, -340, 'Pause')
    vsg.vsgDrawString(0, -300, 'Sie haben %d%% dieses Blocks geschafft!' %progress)
    vsg.vsgDrawString(0, -220, 'Um weiterzumachen, druecken Sie eine Taste.')
    vsg.vsgSetDisplayPage(page_number)
    vsg.cbboxFlush()


def wait_for_button(vsg):
    """
    delay program until any button has been pressed
    """
    theStatus,C = vsg.cbboxCheck()
    while theStatus == vsg_client.respEMPTY:
        theStatus,C = vsg.cbboxCheck()
        time.sleep(0.2)


def get_response(vsg, sleep = 0.2):
    """
    
    """
    theStatus,C = vsg.cbboxCheck()
    while theStatus == vsg_client.respEMPTY:
        theStatus,C = vsg.cbboxCheck()
        # read the last buffered state of the buttons
        # respones sampling rate
        time.sleep(sleep)
    
    rt = C.counter / 1e3
    return vsg_client.getbutton(C), rt


def create_logfile(id, res_dir, var_names):
    results_path = '%s\\%s\\' %(res_dir, id)
    # maske '?<=' - vor  '$' - end of string '+' mind ein und mehr als ein Zeichen z.B. 12
    mask = re.compile('(?<=_)\d+$')
    
    if not os.access(results_path, os.F_OK):
        # check whether results folder exists
        print 'failed to access', results_path
        sys.exit()
    else:
        # if folder exists get current session number
        dirlist = os.listdir(results_path)
        maxnum = 0
        for fname in dirlist:
            found = mask.search(fname)
            if found:
                num = int(found.group())
                if num > maxnum: maxnum = num
        maxnum += 1
        fname = '%s_%d' % (id, maxnum)
        fid = open(results_path + fname, 'w')
        header = ''
        for k in var_names[:-1]:
            header = header + k + '\t'
        header = header + var_names[-1] + '\n'
        fid.write(header)
        return fid


# set timing
# ====================
fix_time  = 0.76 # 1.0

# number of trials
# =====================
radius = 300

# set luminance range
# ====================
mean_lum   = 0.2
contrast   = 0.3
background = 255

### VSG ###
vsg = vsg_client.VsgClient()
vsg.cbboxOpen(vsg_client.respCEDRUS)

LUT_range = np.linspace(mean_lum-contrast/2., mean_lum+contrast/2., 256)
LUT_range = LUT_range.tolist()

vsg.vsgLUTBUFFERWrite(1, LUT_range)
vsg.vsgLUTBUFFERtoPalette(1)

vsg.vsgSetPen1(1)
vsg.vsgSetPen2(background)


# log file name and location
# ============================
id  = raw_input ('Bitte geben Sie Ihre Initialen ein (z.B. mm): ')
fid = create_logfile(id, 'results', ['blk', 'trl', 'target', 'search', 'nitems', 'rot_offset', 'tpos_x', 'tpos_y', 'rt', 'button'])

# design
# ========
# the design matrix specifying target presence, target identity and the number of distractors will be loaded and randomized anew in each block of trials for one of the stimulus types
design = np.loadtxt('design.txt', skiprows=1)

# blocked condition
# ====================
# these correspond to the different types of stimuli, in the new experiment a new category "noinducer" will be included
conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}

stim_in_prev_trl = 5

for trl in design[:,0]:
    """
    start experimental trials - conditions read from design matrix
    """
    trl_nr, curr_search, curr_stim, curr_nitems, curr_target = design[trl,:]
    stimulus_type = conditions[curr_stim]
    
    if curr_stim != stim_in_prev_trl:
        """
        if stimulus condition has changed --> show intro screen
        """
        stim_in_prev_trl = curr_stim
        # show INTRO because stimulus condition changed
        positions, dum = get_positions(random.randint(2,3)*4, radius)
        if random.randint(0,1) == 1:
            show_display_intro(vsg, 5, background, 'fat_%s_10' %stimulus_type, 'thin_%s_10' %stimulus_type, positions)
        else:
            show_display_intro(vsg, 5, background, 'thin_%s_10' %stimulus_type, 'fat_%s_10' %stimulus_type, positions)
        wait_for_button(vsg)
        
        """
        start with trial - fixation
        """
        show_fixation(vsg, 2, background, (0, 0))
        time.sleep(fix_time)
        
        """
        assign target if it is a target present trial
        """
        if curr_target == 1:
            target_id     = 'fat_%s_10' %stimulus_type
            distractor_id = 'thin_%s_10' %stimulus_type
        elif curr_target == 2:
            target_id     = 'thin_%s_10' %stimulus_type
            distractor_id = 'fat_%s_10' %stimulus_type
        
        positions, curr_off = get_positions(curr_nitems, radius)
        targ_x, targ_y      = show_display(vsg, 1, background, curr_search, target_id, distractor_id, positions)
        
        """
        measure response time after stimulus presentation
        """
        vsg.vsgResetTimer()
        button, rt = get_response(vsg)
        
        if button == 0:
            fid.close()
            vsg.vsgClearPage()
            sys.exit()
        else:
            fid.write('%d\t%d\t%d\t%d\t%d\t%d\t%2.2f\t%2.2f\t%2.4f\t%d\n' % (trl, curr_stim, curr_search, curr_nitems, curr_off, targ_x, targ_y, rt, button))

        if (trl+1) % (repeats * 3) == 0 and (trl+1) != (12 * repeats):
            progress = np.round(float(trl+1)/ntrials * 100)
            show_pause(vsg, 4, background, progress)
            wait_for_button(vsg)


vsg.vsgClearPage()
