#!/usr/bin/env python

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.style']  = 'italic'
plt.rcParams['font.size']   = 14
plt.rcParams['lines.markeredgewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.linewidth']  = 2


def hits_theory(d_prime, criterion):
    """
    for distribution with X~N(d_prime,1) and criterion, how many values are > criterion
    """
    return 1-stats.norm.cdf(criterion-d_prime)

def false_alarms_theory(criterion):
    """
    for distribution with X~N(0,1) and criterion, how many values are > criterion
    """
    return 1-stats.norm.cdf( criterion )

def plot_signal_noise_criterion(d_prime, criterion, my_col):
    """
    generate probability distribution functions for noise ~ N(0,1) and signal distribution ~N(d_prime, 1))
    input
    -----
    d_prime
    criterion
    output
    ------
    plot
    """
    
    x = np.linspace(-4, 4, 100)
    pdf_noise = stats.norm.pdf(x, 0, 1)
    
    plt.plot(x,           pdf_noise, '-.', color = my_col)
    plt.plot(x + d_prime, pdf_noise, '-',  color = my_col)
    
    plt.plot([0,0], [0, 0.4], 'k-')
    plt.plot([d_prime]   * 2, [0, 0.4], 'k-')
    plt.plot([criterion] * 2, [0, 0.4], 'k-')


if __name__ == '__main__':
    
    colors = {0.25: 'red', 0.5: 'blue', 1: 'orange', 3:'green', 7: 'cyan'}
    f1 = plt.figure()
    plt.show()

    for idx, d_prime in enumerate(np.sort(colors.keys())):
        hits = []
        fas  = []
        for criterion in np.linspace(0, d_prime, 5):
            curr_hit = hits_theory(d_prime, criterion)
            curr_fa  = false_alarms_theory(criterion)
            hits.append(curr_hit)
            fas.append(curr_fa)
            
            plt.subplot(3, 2, (idx+1))
            plot_signal_noise_criterion(d_prime, criterion, colors[d_prime])
            plt.title('d\' = %2.2f' %d_prime)
            plt.xlim([-4, 11])
            
            plt.subplot(3, 2, 6)
            plt.plot(curr_fa, curr_hit, 'o', color = colors[d_prime])
            plt.plot([0,1], [0,1], 'k-')
            plt.xlim([0,1])
            plt.ylim([0,1])
            #raw_input('press return to continue')

#f1_width, f1_height = f1.get_size_inches()
#f1.set_size_inches([f1_width, f1_width/f1_height * f1_height])
#plt.savefig('')


