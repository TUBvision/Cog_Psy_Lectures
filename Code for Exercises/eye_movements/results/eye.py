import numpy as np
import sys
import os

__all__ = ['get_trials', 'raw_to_trial', 'samples_to_ms', 'samples_to_ms_coord', 'transform_coord', 'data_to_dict', 'get_subset', 'dist', 'veloc_to_fix', 'velocity_based_identification', 'dispersion_based_identification',  'fix_index_to_fixations']

def get_trials(data, start, end):
    """
    Formatting function which splits the data into trials and removes the header.
    
    Variables
    ----------
    data    - string
    start
    end     - trial identifier strings
    
    Returns
    ----------
    trials  - a list of trials
    """
    if data.find(start) == -1:
        print 'wrong trial identifier, try other than %s' %start
    trial_start = data.split(start)
    trials      = [ trl[:trl.index(end)] for trl in trial_start[1:] ]
    return trials


def raw_to_trial(trial):
    """
    extract from trial time, xpos, ypos; add trl_id and blk_id
    """
    trl_events  = trial.split("\n")
    trl_number  = np.int(trl_events[0].strip('_'))
    trl_samples = trl_events[1:-1]
    trl_samples = [t.split('\t') for t in trl_samples]
    trl_samples = [[trl_number, np.float(t[0]), np.float(t[1]), np.float(t[2])] for t in trl_samples if t[1].strip() != '.']
    return np.array(trl_samples)


def samples_to_ms(samples):
    """
    Zeit transformieren
    """
    time = samples/1000.
    time = time-time[0]
    return time


def transform_coord(vector, position):
    """
    from [0,0] lower left to [0,0] in center
    """
    new_pos = vector-position
    return new_pos


def samples_to_ms_coord(trial_data, xpos, ypos):
    """
    """
    trial_data[:,0] = samples_to_ms(trial_data[:,0])
    trial_data[:,1] = transform_coord(trial_data[:,1], xpos)
    trial_data[:,2] = transform_coord(trial_data[:,2], ypos)
    return trial_data


def data_to_dict(fname):
    """
    read tab separated data columns with headerline from logfile into dictionary
    :input:
    ------
    filename
    :output:
    ------
    dictionary
    """
    if not os.path.isfile(fname):
        data_frame = 0
        print 'file does not exist'
    else:
        header = open(fname, 'r').readline().strip('\n\r').split('\t')
        data   = np.loadtxt(fname, skiprows = 1)
        data_frame = {}
        for var, var_name in enumerate(header):
            data_frame[var_name] = data[:,var]
    return data_frame


def get_subset(data, subset_variable, subset_value, subset_operator='eq'):
    """
    get subset of dictionary according to condition in one variable
    :input:
    --------
    data            - original dictionary
    subset_variable - key of variable
    subset_value    - value of condition
    subset_operator eq(=default), ue - unequal, gt - greater than, st - smaller than
    """
    sub_data = {}
    for var in data.keys():
        if subset_operator == 'eq':
            sub_data[var] = data[var][data[subset_variable] == subset_value]
        elif subset_operator == 'ue':
            sub_data[var] = data[var][data[subset_variable] != subset_value]
        elif subset_operator == 'gt':
            sub_data[var] = data[var][data[subset_variable] > subset_value]
        elif subset_operator == 'st':
            sub_data[var] = data[var][data[subset_variable] < subset_value]
    return sub_data


def millisecs_to_nsamples(ms, sampling_rate):
    """
    convert duration threshold in ms into duration threshold in number of sampling points
    input:
    ------
    minimal fixation duration in ms
    sampling_rate in number of samples per second
    output
    ------
    minimal fixation duration in number of samples
    """
    return np.int(ms * sampling_rate/1000.)


def del_artefact(currwindow, xlim=640, ylim=512):
    """
    artefact removal based on geometry of screen,
    excludes x- and y-positions that are outside the screen coordinates
    :Input:
    currwindow      3 columns of raw data
    xlim            limits in x- and y-direction
    ylim
    
    :Output:
    currwindow      3 columns of raw data with outliers removed
    """
    x = currwindow[:,0]
    y = currwindow[:,1]
    xdir = np.logical_and(np.greater(x, -xlim), np.less(x, xlim))
    ydir = np.logical_and(np.greater(y, -ylim), np.less(y, ylim))
    valid_idx = np.logical_and(xdir, ydir)
    return currwindow[valid_idx,:]


def calc_veloc(raw):
    """
    calculate velocity between sampling points as euclid distance
    :Input:
    raw     - contains x and y positions (2 columns)
    """
    veloc=[]
    for k in range(len(raw)-1):
        # euclidean distance
        ed =  dist(raw[k,:], raw[k+1,:])
        veloc.append(ed)
    return np.array(veloc)


def moving_average(velocity_samples, sampling_rate, window_size=None, mode=None):
    """
    smoothing of velocity samples with rectangular window of window_size
    :input:
    -------
    velocity_samples    difference vector (n-1) of original samples (n)
    sampling_rate
    window_size         input in ms, smooth_window is converted into samples
    mode                one of 'full', 'valid', 'same'
    :output:
    --------
    smoothed velocity
    """
    smooth_window = round(window_size * sampling_rate/1000)
    window = np.ones(smooth_window)
    return np.convolve(velocity_samples, window, mode)/smooth_window


def pixel_to_degree(size_pixel, screen_pixel=[1920., 1200.], screen_mm=[518., 324.], view_dist=600.):
    """
    convert pixel into degrees of visual angle
    
    :Input:
    pixel   size in pixels
    screen_pixel
    screen_mm
    view_dist in mm
    
    :Output:
    alpha       angle in degrees
    """
    alpha_mm = size_pixel/(screen_pixel[0]/screen_mm[0])
    alpha    = np.degrees(2 * np.arctan2( alpha_mm, 2 * view_dist))
    return alpha


def pixel_to_mm(x, screen_pixel=1920., screen_mm=518.):
    """
    :input:
    x               size in pixel
    screen_pixel    screen size in pixel
    screen_mm       screen size in mm
    :output:
    x               size in mm
    """
    return screen_mm/screen_pixel * x


def mm_to_visangle(x, view_distance=600.):
    """
    :input:
    x               size in mm
    view_distance   in mm
    :output:
    x               size in degree visual angle
    """
    return np.degrees(2 * np.arctan2(x, 2 * view_distance))


def dist(x,y):
    """
    compute euclidean distance between two points
    :input:
    
    """
    return np.sqrt(np.sum((x-y)**2))


def veloc_to_fix(velocity, velocity_threshold, duration_threshold, sampling_rate):
    """
    velocity based fixation identification
    fixations < velocity_threshold & > duration_threshold
    :Input:
    -------
    velocity            in degrees visual angle
    velocity_threshold  in degrees visual angle/sec [20 degress/sec Sen & Megaw]
    duration_threshold  in ms
    sampling_rate
    :Output:
    --------
    indices of fixations
    """
    nsamples = len(velocity)
    
    fix_velocities = np.array(velocity < velocity_threshold, dtype = int)
    
    #raw_fix_mean = np.mean(velocity[fix_velocities==1])
    #raw_fix_sd   = np.std(velocity[fix_velocities==1])
    
    #if velocity_threshold <= raw_fix_mean + 2 * raw_fix_sd:
        #velocity_threshold = raw_fix_mean + 2 * raw_fix_sd
        #fix_velocities = np.array(velocity < velocity_threshold, dtype = int)
    
    # ... after number of occurrence
    fixation_index = np.diff(fix_velocities)
    fix_start = np.where(fixation_index == 1)[0]+1
    # assumption that eye movements start with fixation
    fix_start = np.r_[0, fix_start]
    # add 1 index because difference-vector index shifted by one element
    fix_end   = np.nonzero(fixation_index == -1)[0]+1
    fix_end   = np.r_[fix_end, nsamples]
    
    if fix_velocities[-1] == 0:
        fix_end = fix_end[:-1]
    
    if fix_velocities[0] == 0:
        fix_start = fix_start[1:]
    
    fix_index = np.c_[fix_start, fix_end]
    
    # eliminate fixations with DUR < dur_crit ms
    critical_duration = duration_threshold * sampling_rate/1000
    fix_dur   = fix_index[:,1]-fix_index[:,0]
    fix_index = fix_index[fix_dur>critical_duration,:]
    
    return fix_index


def velocity_based_identification(data_samples, velocity_threshold, duration_threshold,  smooth_window, sampling_rate=1000.):
    """
    :input:
    --------
    data_samples        x- and y-pos of each sample in deg visual angle
    sampling_rate       number of samples/second
    velocity_threshold  velocity in deg/sec < fixations
    duration_threshold  minimum duration of a fixation
    smooth_window       size of 
    """
    velocity = calc_veloc(data_samples * sampling_rate)
    
    # smooth raw velocities with smoothing window
    velocity_smooth = moving_average(velocity, sampling_rate, smooth_window, 'same')
    
    fix_index = veloc_to_fix(velocity_smooth, velocity_threshold, duration_threshold, sampling_rate)
    
    fixations = fix_index_to_fixations(data_samples, fix_index, sampling_rate)
    return fixations


def dispersion_based_identification(data_samples, dispersion_threshold, duration_threshold, sampling_rate=1000.):
    """
    compute fixations based on dispersion of sample points and minimum duration, disperson = mean(max_horizontal, max_vertical distance)
    :input:
    --------
    data_samples
    dispersion_threshold        in degree visual angle
    duration_threshold          in ms
    :output:
    --------
    
    """
    data_samples_or = data_samples.copy()
    duration_threshold_in = millisecs_to_nsamples(duration_threshold, sampling_rate)
    sacc_samples = 0
    fix_end   = 0
    fix_index = []
    
    while data_samples.shape[0] >= duration_threshold_in:
    
        duration_samples = duration_threshold_in
        window    = data_samples[:duration_samples,:]
    
        ### calculate distance as radius
        #centroid  = np.mean(window, 0)
        #distances = np.array([dist(centroid, window[k,:]) for k in range(window.shape[0])])
    
        ## calculate maximal horizontal and vertical distances
        d_x = np.abs(np.max(window[:,0])-np.min(window[:,0]))
        d_y = np.abs(np.max(window[:,1])-np.min(window[:,1]))
        distances = np.array([d_x, d_y])
    
        if np.mean(distances) <= dispersion_threshold:
        
            while np.mean(distances) <= dispersion_threshold:
                duration_samples +=1
                if duration_samples > data_samples.shape[0]:
                    break
                window    = data_samples[:duration_samples,:]
                ## calculate distance
                d_x = np.abs(np.max(window[:,0])-np.min(window[:,0]))
                d_y = np.abs(np.max(window[:,1])-np.min(window[:,1]))
                distances = np.array([d_x, d_y])
            
            fix_start = fix_end + sacc_samples
            fix_end   = fix_start + (duration_samples-1)
            sacc_samples = 0
            fix_index.append([fix_start, fix_end])
            
            data_samples = data_samples.take(np.arange(duration_samples, data_samples.shape[0]), axis=0)
        else:
            data_samples = data_samples.take(np.arange(1, data_samples.shape[0]), axis=0)
            sacc_samples +=1
    
    fixations = fix_index_to_fixations(data_samples_or, np.array(fix_index), sampling_rate)
    return fixations


def fix_index_to_fixations(data_samples, fix_index, sampling_rate=1000.):
    """
    """
    n_fix  = fix_index.shape[0]
    if n_fix == 0:
        fixations = nans((1,5))
    else:
        f_dur  = np.diff(fix_index) * (1000 / sampling_rate)
        s_dur  = (fix_index[1:,0] - fix_index[:-1,1]) * (1000 / sampling_rate)
        x_pos = [np.mean(data_samples[fix_index[k,0] :fix_index[k,1], 0]) for k in range(n_fix)]
        y_pos = [np.mean(data_samples[fix_index[k,0] :fix_index[k,1], 1]) for k in range(n_fix)]
        
        fixations = np.c_[range(n_fix), f_dur, np.r_[s_dur, np.nan],  x_pos, y_pos]
    return fixations


def nans(shape, dtype=float):
    """
    create numpy array of size (r,c) containing NaN
    """
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a
