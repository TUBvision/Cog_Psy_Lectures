import numpy as np
import sys
import os
import eye

id   = sys.argv[1]
sess = np.int(sys.argv[2])

#id   = 'mm'
#sess = 2

# read raw data files into string variable d
d = open('%s/%s_%d.asc' %(id, id, sess)).read()

# separate raw data into list of trials
raw_trials = eye.get_trials(d, 'trial_start', 'BUTTON')

all_trials = []

for trl_nr, raw_trial in enumerate(raw_trials):
    print trl_nr
    # single trial conversion list into numpy array
    trial_data = eye.raw_to_trial(raw_trial)
    
    # single trial conversion of samples into timesteps and screen coordinates
    trial_data[:,1:] = eye.samples_to_ms_coord(trial_data[:,1:], 960, 600)
    
    all_trials.append(trial_data)

all_trials = np.concatenate(all_trials, axis=0)

# write dataout in text file
out = open('%s/%s_%d_eye.txt' %(id, id, sess), 'w')
out.write('blk time xpos ypos\n')
np.savetxt(out, all_trials, fmt='%d %1.5f %1.2f %1.2f', delimiter='\t')
out.flush()
out.close()


