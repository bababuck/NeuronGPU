import numpy as np
import scipy

if __main__:
    for i in np.arange(len(6))[::-1]:
        rates_per_neuron_rev = \
            np.loadtxt(f'rate{i}.dat')

        scipy.stats.entropy(
