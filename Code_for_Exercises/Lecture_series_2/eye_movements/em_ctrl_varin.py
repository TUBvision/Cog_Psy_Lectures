#!/usr/bin/env python
#coding: utf-8

import re
import sys
import time

import Image, ImageDraw, ImageFont

#eyelink
from pylink import *
from EyeLinkCoreGraphicsPyGame import EyeLinkCoreGraphicsPyGame
from pygame import display

from hrl import HRL
import numpy as np
from random import uniform

from search_for_filled import *



def calibrate_eyelink(screen_width, screen_height, stim_width, stim_height):
    #warning: vsg must be global
    ## CALIBRATION
    eyelink = getEYELINK()

    pylink.flushGetkeyQueue(); 
    eyelink.setOfflineMode();                         
    #Gets the display surface and sends a mesage to EDF file;
    surf = display.get_surface() # doesn't work under Linux+OpenGL, returns None
    #screen_width = surf.get_width()
    #screen_height = surf.get_height()
    print 'get surface!'
    # left = vsg.width/2 - stim_width/2
    # top = vsg.height/2 - stim_height/2
    # right = left+stim_width
    # bottom = top+stim_height
    left = screen_width/2 - stim_width/2
    top = screen_height/2 - stim_height/2
    right = left+stim_width
    bottom = top+stim_height

    print 'left: %d top: %d right: %d bottom: %d' % (left,top,right,bottom)
    eyelink.sendCommand("screen_pixel_coords = %d %d %d %d" % (left,top,right,bottom))
    eyelink.sendMessage("DISPLAY_COORDS  %d %d %d %d" %(left,top,right,bottom))
    eyelink.setCalibrationType("HV9") #to recalculate the coordinates for targets

    dispinfo = getDisplayInformation()
    

    if(eyelink.isConnected() and not eyelink.breakPressed()):
            print "Starting trial function"

    print 'starting tracker setup'
    eyelink.doTrackerSetup()        #ESC beendet die Kalibrierung
    print 'setup ready'
    msg = eyelink.getCalibrationMessage()
    print 'message after calibration:', msg
    fl = open('calibration.txt','a')
    if len(sys.argv) == 2:
        vp_id = sys.argv[1]
    else:
        vp_id = 'secret'
    msg = msg.split()
    if len(msg) >= 5:

        left_avg = float(msg[1])
        right_avg = float(msg[4])
        print 'Average error left: %f right %f' % (left_avg,right_avg)

        fl.write("%s %.2f %.2f %s\r\n" % (time.strftime('%d.%m.%y %H:%M'),left_avg,right_avg,vp_id))
    elif len(msg) >= 2:
        left_avg = float(msg[1])
        fl.write("%s %.2f %s\r\n" % (time.strftime('%d.%m.%y %H:%M'),left_avg,vp_id))
    fl.close()


def clear_queue():
    eyelink = getEYELINK()
    num = eyelink.getDataCount(1,1)
    for i in range(num):
        eyelink.getNextData()
    print 'wiped %d items from the queue' % num


def stop_trial():
    print 'stopping'
    endRealTimeMode()
    eyelink = getEYELINK()
    eyelink.stopRecording()
    print 'receiving EDF...'
    eyelink.closeDataFile()
    eyelink.receiveDataFile(edfFileName, results_path + edfFileName)
    #closeGraphics()
    #sys.exit()
    print 'done'

def image2np(img):
    """
    Converts an Image object into numpy array with values between 0 and 1

    Parameters
    ----------
    img: instance of Image() or a filename as string
    """

    if type(img) == str:
        img = Image.open(img)
    buf = np.array(img, dtype=np.float)
    buf /= 255.
    return buf


def draw_display(hrl, page_number, background, target, target_id, distractor_id, positions):
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
    nitems = len(positions['x'])
    target_pos = random.randint(0, nitems-1)
    targ_x, targ_y = 0,0
    for k in np.arange(nitems):
        x, y = positions['x'][k], positions['y'][k]
        fname = 'thin_misaligned_10.bmp'
        if target_pos == k and target == 1:
            targ_x, targ_y = x, y
            mnptch = hrl.newTexture(image2np('stimuli/%s.bmp' %target_id), "square")
            mnptch.draw((targ_x, targ_y), (111, 111))
            #mnptch.draw((targ_x, targ_y))

        else:
            mnptch = hrl.newTexture(image2np('stimuli/%s.bmp' %distractor_id), "square")
            mnptch.draw((x, y), (111, 111))


    return (targ_x, targ_y)


def draw_display_pause(hrl, progress):
    im = Image.new('L', (1000, 800), background)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", 48)
    draw.text((0,0), "Pause", font=font)
    draw.text((0,50), "Sie haben %d%% geschafft" % progress, font=font)
    draw.text((0,100), "Um weiterzumachen, druecken Sie eine Taste.", font=font)
    pauseptch = hrl.newTexture(image2np(im), "square")
    pauseptch.draw((0, -340))
    hrl.flip(clr=True)

def draw_display_intro(hrl):
    im = Image.new('L', (620, 400), background)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", 30)
    draw.text((0,0), "So werden die naechsten Aufgaben aussehen", font=font)
    draw.text((0,50), "Schauen Sie alle Figuren an", font=font)
    draw.text((0,100), "und druecken Sie danach eine Taste", font=font)
    introptch = hrl.newTexture(image2np(im), "square")
    introptch.draw((-600, -540))
    hrl.flip(clr=True)


def exit_menu():
    print '[r] to recalibrate EyeLink'
    print '[c] to continue experiment'
    print '[e] to exit'
    r = raw_input('->')
    if r == 'r':
        pylink.openGraphics()
        calibrate_eyelink(screen_width, screen_height, stim_width, stim_height)
        pylink.closeGraphics()
        return r
    elif r == 'c':
        print 'continue experiment'
        return r
    elif r == 'e':
        print 'interrupted by user'
        return r

    

def wait_fixation(hrl, pos, delay, clr=True):
    active = False
    eyelink = getEYELINK()
    #eyelink = EyeLinkListener()
    print 'waiting fixation on:', pos
    cross_np = image2np('fixation_cross.bmp')
    cross_np_min = cross_np.min()
    cross_active_np = cross_np.copy()
    cross_active_np[cross_active_np == cross_np_min] = 0.75
    cross_width = cross_height = 200
    cross_passive = hrl.newTexture(cross_np)
    cross_passive.draw((0, 0), (cross_width, cross_height))
    cross_active = hrl.newTexture(cross_active_np)
    print 'Clear:', clr
    hrl.flip(clr=clr)
    startfix = 0
    while 1:
        new_data = eyelink.getNextData()
        #new_data = eyelink.getLastData()
        if new_data:
                eye_event = eyelink.getFloatData()
                if eye_event:
                        event_type = eye_event.getType()
                        #print 'event_type:', event_type
                        if event_type != 200:
                                if event_type == STARTFIX:
                                    if not active:
                                        startfix = time.time()
                                
                                        #print "Fixation start"
                                elif event_type == FIXUPDATE:
                                        #print "Fixation update"
                                        eye_x,eye_y = eye_event.getAverageGaze()
                                        #print "average gaze: %s" % list(eye_event.getAverageGaze())
                                        #print 'drift:', abs(eye_x-pos[0]), abs(eye_y-pos[1])
                                        if abs(eye_x-pos[0]) < margin and abs(eye_y-pos[1]) < margin:
                                            if startfix == 0: startfix = time.time()
                                            else:
                                                #print "I see! I see! The patient is looking at me!"
                                                #active cross
                                                if not active:
                                                    cross_active.draw((0,0), (cross_width,cross_height))
                                                    hrl.flip(clr=clr)
                                                    #vsg.vsgSetPen1(0);
                                                    #vsg.vsgDrawString(0,30,'+')
                                                    active = True

                                                if time.time() - startfix >= delay:
                                                    #print 'Long enough!', time.time() - startfix
                                                    #print '====================================\r\n\r\n'
                                                    break
                                                #else:
                                                #    print 'But too short!', time.time() - startfix
                                        else:
                                            if active:
                                                cross_passive.draw((0,0), (cross_width,cross_height))
                                                hrl.flip(clr=clr)
                                                #vsg.vsgSetPen1(100);
                                                #vsg.vsgDrawString(0,30,'+')
                                                active = False
                                                startfix = 0



def get_response():
    """
    Get response from button box (joystick)
    returns:
           int button .. code of button
           int rt .. reaction time in seconds
    """
    eyelink = getEYELINK()
    eyelink.flushKeybuttons(0) # 0 .. don't queue press events
    start = eyelink.trackerTime()
    while 1:
        button, timestamp = eyelink.getLastButtonPress()
        if button:
            #rt = (timestamp - start) / 1e3 # in seconds
            rt = (timestamp - start) # in milliseconds
            return button, rt
        else:
            time.sleep(0.1)

def get_last_trial_number(fname):
    fl = open(results_path + fname)
    lines = fl.readlines()
    last_line = lines[-1]
    blk = int(last_line.split()[0])
    trl_nr = int(last_line.split()[1])
    return blk, trl_nr

def open_logfile(id, res_dir, var_names):
    #jetzt global: results_path = '%s/%s/' %(res_dir, id)
    # maske '?<=' - vor  '$' - end of string '+' mind ein und mehr als ein Zeichen z.B. 12
    mask = re.compile('(?<=_)\d+$')
    
    if not os.access(results_path, os.F_OK):
        # check whether results folder exists
        print 'failed to access', results_path
        try:
            print 'trying to create results folder'
            #os.mkdir(results_path)
            os.makedirs(results_path) # will create intermediate directories also
        except:
            print 'failed to create folder, exiting...'
            sys.exit()
    if os.access(results_path, os.F_OK):
    #else:
        # if folder exists get current session number
        dirlist = os.listdir(results_path)
        maxnum = 0
        for fname in dirlist:
            found = mask.search(fname)
            if found:
                num = int(found.group())
                if num > maxnum: maxnum = num

        if maxnum:
            last_blk, last_trial = get_last_trial_number("%s_%d" % (id, maxnum))
            blk = last_blk + 1
        else:
            last_trial = None
            last_blk = 0
            blk = 1

        if last_trial:
            print 'Last block: %d\tLast trial: %d' % (last_blk, last_trial)
        else:
            print 'First trial for this subject'
        maxnum += 1
        fname = '%s_%d' % (id, maxnum)
        fid = open(results_path + fname, 'w')
        header = ''
        for k in var_names[:-1]:
            header = header + k + '\t'
        header = header + var_names[-1] + '\n'
        fid.write(header)
        # Opens the EDF file.
        edfFileName = "%s_%d.EDF" % (id.upper(), maxnum)
        eyelink.openDataFile(edfFileName)		

        return fid, edfFileName, blk, last_trial

def get_ntrials(fname):
    """
    Returns number of trials for given design matrix

    fname .. name of file containing design matrix
    """
    fl = open(fname)
    last_line = fl.readlines()[-1]
    ntrials = int(last_line.split()[0])
    return ntrials

def show_intro(hrl, design, conditions, trl_line):
            trl_b, curr_stim, curr_search, curr_nitems, curr_target = design[trl_line,:] # read next line
            stimulus_type = conditions[curr_stim]

            if curr_target == 1:
                target_id     = 'fat_%s_10' %stimulus_type 
                distractor_id = 'thin_%s_10' %stimulus_type
            elif curr_target == 2:
                target_id     = 'thin_%s_10' %stimulus_type
                distractor_id = 'fat_%s_10' %stimulus_type 

            positions, curr_off = get_positions(8, radius)
            targ_x, targ_y      = draw_display(hrl, 1, background, 1, target_id, distractor_id, positions)
            draw_display_intro(hrl)
            button, rt = get_response()

    

def main(blk, last_trial):
    # HRL parameters
    # wdth = 1024
    # hght = 768
    wdth = 1920
    hght = 1200


    # Section the screen - used by Core Loop
    wqtr = wdth/4.0
    
    # IO Stuff
    dpxBool = False
    dfl = 'design.txt'
    rfl = 'results/res.csv'
    flds = ['TrialTime','SelectedLuminance']
    btns = ['Yellow','Red','Blue','Green','White']
    #ntrials = get_ntrials(dfl)
    

    # Central Coordinates (the origin of the graphics buffers is at the centre of the
    # screen. Change this if you don't want a central coordinate system. If you delete
    # this part the default will be a matrix style coordinate system.
    #coords = (0,1,1,0)
    coords=(-0.5,0.5,-0.5,0.5)
    flipcoords = False

    # Pass this to HRL if we want to use gamma correction.
    lut = 'LUT.txt'
    # If fs is true, we must provide a way to exit with e.g. checkEscape().
    fs = True # set to False while debugging, else the mouse cursor sometimes will be trapped in the second screen

    # Step sizes for luminance changes
    smlstp = 0.01
    bgstp = 0.1

    bg = background/255.
    # HRL Init
    hrl = HRL(wdth,hght,dpx=dpxBool,dfl=dfl,rfl=rfl,rhds=flds
              ,btns=btns,fs=fs,coords=coords,flipcoords=flipcoords,bg=bg)





    # Draw but don't clear the back buffer
    hrl.flip(clr=True)

    # Prepare Core Loop logic
    btn = None
    t = 0.0

    #TODO: show_intro()



    conditions = {0: 'misaligned', 1: 'aligned', 2: 'filled', 3: 'noinducer'}

    design = np.loadtxt('design/em_control.txt', skiprows=1)

    
    ntrials = design.shape[0]

    # show first stimulus type for the subject, before the experiment begins
    show_intro(hrl, design, conditions, 0)
    eyelink.startRecording(1,1,1,1) # file samples, file events, link samples, link events
    for trl_line in design:
        """
        start experimental trials - conditions read from design matrix
        """
        #trl_line_n += 1 # number of current line in design matrix
        try:
            trl, curr_stim, curr_search, curr_nitems, curr_target = trl_line
        except IndexError:
            print 'Index failed:', trl_line
            break # dirty
        #trl = int(dsgn['trl_n'])

        continue_trial = True # continue trial after abort
        if last_trial and continue_trial:
            if trl != last_trial: 
                print 'skipping trial: %d' % trl
                continue
            else:
                print 'last trial found: %d' % trl
                last_trial = None
                continue # last skip

        print trl, curr_search, curr_stim, curr_nitems, curr_target
        stimulus_type = conditions[curr_stim]

        """
        assign target if it is a target present trial
        """
        if curr_target == 1:
            target_id     = 'fat_%s_10' %stimulus_type 
            distractor_id = 'thin_%s_10' %stimulus_type
        elif curr_target == 2:
            target_id     = 'thin_%s_10' %stimulus_type
            distractor_id = 'fat_%s_10' %stimulus_type 
        
        fix_time = np.random.normal(0.75, 0.25) # mu = 0.75, sigma = 0.25
        wait_fixation(hrl, (screen_width/2,screen_height/2), fix_time)

        
        positions, curr_off = get_positions(curr_nitems, radius)
        #targ_x, targ_y      = show_display(vsg, 1, background, curr_search, target_id, distractor_id, positions)
        targ_x, targ_y      = draw_display(hrl, 1, background, curr_search, target_id, distractor_id, positions)
        start_of_trial = time.time()
        eyelink.sendMessage("TRIALID %d" % trl)
        eyelink.sendMessage("trial_start_%d" % trl)

        hrl.flip(clr=True)
        #time.sleep(1.5)
        #wait_fixation(hrl, (screen_width/2,screen_height/2), 2.25, False)
        #rt = time.time() - start_of_trial
        button, rt = get_response()
        if button == 6: print 'No'
        elif button == 7: print 'Yes'
        elif button == 5:
            stop_trial()
            #eyelink.stopRecording()
            break
        if hrl.checkEscape(): break
        button = 6 # target never present
        eyelink.sendMessage("TRIAL_RESULT %d" % button)
        eyelink.sendMessage("trial_end_%d" % trl)

        #fid.write('%d\t%d\t%d\t%d\t%d\t%d\t%2.2f\t%2.2f\t%2.4f\t%d\n' % (blk, trl, curr_stim, curr_search, curr_nitems, curr_off, targ_x, targ_y, rt, button)) # rt in seconds
        fid.write('%d\t%d\t%d\t%d\t%d\t%d\t%2.2f\t%2.2f\t%d\t%d\n' % (blk, trl, curr_stim, curr_search, curr_nitems, curr_off, targ_x, targ_y, rt, button)) # rt in milliseconds

        if (not trl % 60) and (trl != ntrials): # pause every 60 trials
            print 'pause'
            eyelink.stopRecording()
            progress = np.round(float(trl)/ntrials * 100)
            draw_display_pause(hrl, progress)
            button, rt = get_response()
            if button == 2:
                # TODO: cannot recalibrate while hrl open
                hrl.close() # close HRL context for EyeTracker calibration

                res = exit_menu()
                if res == 'e':
                    break

                hrl = HRL(wdth,hght,dpx=dpxBool,dfl=dfl,rfl=rfl,rhds=flds
                          ,btns=btns,fs=fs,coords=coords,flipcoords=flipcoords,bg=bg)
                hrl.flip(clr=True) # return to HRL context

            # show example of next trial after pause

            trl_b, curr_stim, curr_search, curr_nitems, curr_target = design[trl_line[0]+1,:] # read next line
            stimulus_type = conditions[curr_stim]

            if curr_target == 1:
                target_id     = 'fat_%s_10' %stimulus_type 
                distractor_id = 'thin_%s_10' %stimulus_type
            elif curr_target == 2:
                target_id     = 'thin_%s_10' %stimulus_type
                distractor_id = 'fat_%s_10' %stimulus_type 

            positions, curr_off = get_positions(8, radius)
            targ_x, targ_y      = draw_display(hrl, 1, background, 1, target_id, distractor_id, positions)
            draw_display_intro(hrl)
            button, rt = get_response()

            #
                
            eyelink.startRecording(1,1,1,1)
        
        hrl.flip(clr=True)
        time.sleep(0.5)

        

    # Experiment is over!
    hrl.close()
    stop_trial()




############
# Global variables
radius = 400
#fix_time = 0.76 # NEW: calculate randomly for each trial
background = 127
foreground = 0 # colour of calibration targets
margin = 50 # fixation margin


############
# Setup EyeLink
eyelink = EyeLink()
if not eyelink:
    print "EL is None"
    sys.exit()
else:
    eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE")
    eyelink.sendCommand("link_event_data = GAZE,FIXAVG")
    #eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")
    eyelink.sendCommand("link_event_filter  = LEFT,RIGHT,FIXATION,FIXUPDATE,BUTTON")
    #eyelink.sendCommand("link_event_filter  = FIXATION,FIXUPDATE")
    eyelink.sendCommand("fixation_update_accumulate = 2000");
    eyelink.sendCommand("binocular_enabled = NO")

    #genv = EyeLinkCoreGraphicsPyGame(vsg.width,vsg.height,eyelink,vsg)
    #580x580 is the size of stimulus
    screen_width = 1920
    screen_height = 1200
    stim_width  = 800
    stim_height = 800
    #genv = EyeLinkCoreGraphicsPyGame()    
    #openGraphicsEx(genv)

    # setup logging
    id  = raw_input ('Bitte geben Sie Ihre Initialen ein (z.B. mm): ')
    res_dir = 'results/emc/'
    results_path = '%s/%s/' % (res_dir, id)
    fid, edfFileName, blk, last_trial = open_logfile(id, res_dir, ['blk', 'trl', 'target', 'search', 'nitems', 'rot_offset', 'tpos_x', 'tpos_y', 'rt', 'button'])


    #uncomment this!
    pylink.setCalibrationColors((foreground, foreground, foreground), (background, background, background))
    pylink.openGraphics()
    calibrate_eyelink(screen_width, screen_height, stim_width, stim_height)
    pylink.closeGraphics()
    main(blk, last_trial)
