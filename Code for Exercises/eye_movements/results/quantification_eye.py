import numpy as np
import eye

"""
This script contains functions use to quantify the eye movements in various ways
Namely, extracting fixation points and saccades.

"""

__all__ = [ 'calc_veloc', 'moving_average', 'veloc_to_fix', 'velocity_based_identification','dispersion_based_identification' ]

def calc_veloc(raw):
    """
    calculate velocity between sampling points as euclidian distance
    
    Parameters
    ----------
    raw : 
        contains x and y positions (2 columns)
    
    Returns
    ----------
    veoc : 
    """
    veloc=[]
    for k in range(len(raw)-1):
        # euclidean distance
        ed = np.sqrt(np.sum((raw[k,:]-raw[k+1,:])**2))
         
        veloc.append(ed)
    return np.array(veloc)


def moving_average(velocity_samples, sampling_rate, window_size=None, mode=None):
    """
    smoothing of velocity samples with rectangular window of window_size
    
    Parameters
    ----------
    velocity_samples :
        difference vector (n-1) of original samples (n)
    sampling_rate :
        
    window_size :
    input in ms, smooth_window is converted into samples
    mode :
        one of 'full', 'valid', 'same'
    
    Returns
    ----------
    smoothed_velocity : 
        
    """
    smooth_window = round(window_size * sampling_rate/1000)
    window = np.ones(smooth_window)
    smoothed_velocity = np.convolve(velocity_samples, window, mode)/smooth_window
    return smoothed_velocity


def veloc_to_fix(velocity, velocity_threshold, duration_threshold, sampling_rate):
    """
    velocity based fixation identification
    fixations < velocity_threshold & > duration_threshold
    
    Parameters
    ----------
    velocity :
        in degrees visual angle
    velocity_threshold :
        in degrees visual angle/sec [20 degress/sec Sen & Megaw]
    duration_threshold :
        in ms
    sampling_rate :
        
    
    Returns
    ----------
    fix_index : 
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
    
    
    Parameters
    ----------
    data_samples : 
        x- and y-pos of each sample in deg visual angle
    sampling_rate :
        number of samples/second
    velocity_threshold : 
        velocity in deg/sec < fixations
    duration_threshold :
        minimum duration of a fixation
    smooth_window : 
        size of 
    
    Returns
    ----------
    fixations : 
        
    """
    velocity = calc_veloc(data_samples * sampling_rate)
    
    # smooth raw velocities with smoothing window
    velocity_smooth = moving_average(velocity, sampling_rate, smooth_window, 'same')
    
    fix_index = veloc_to_fix(velocity_smooth, velocity_threshold, duration_threshold, sampling_rate)
    
    fixations = eye.fix_index_to_fixations(data_samples, fix_index, sampling_rate)
    return fixations


def dispersion_based_identification(data_samples, dispersion_threshold, duration_threshold, sampling_rate=1000.):
    """
    compute fixations based on dispersion of sample points and minimum duration, disperson = mean(max_horizontal, max_vertical distance)
    
    Parameters
    ----------
    data_samples : 
        
    dispersion_threshold : 
        in degree visual angle
    duration_threshold : 
        in ms
    
    Returns
    ----------
    fixations :
        
    """
    data_samples_or = data_samples.copy()
    duration_threshold_in = eye.millisecs_to_nsamples(duration_threshold, sampling_rate)
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
    
    fixations = eye.fix_index_to_fixations(data_samples_or, np.array(fix_index), sampling_rate)
    return fixations