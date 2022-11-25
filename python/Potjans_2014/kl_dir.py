import os
import sys
import numpy as np
from scipy.stats import entropy


def get_kl():
    populations =  ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']

    def _get_histogram(population_idx, resolution):
        path = f"./Data{resolution}"
        rates_firing_rates = np.loadtxt(os.path.join(path, ('rate' + str(population_idx) + '.dat')))
        hist, _ = np.histogram(rates_firing_rates,bins=150,range=(0., 50.))
        hist = [i + 1 for i in hist]
        return hist

    def _get_kl(population_idx, resolution):
        pk = _get_histogram(population_idx, resolution)
        qk = _get_histogram(population_idx, 0)
        return entropy(qk, pk)

    def _get_avg_kl(resolution):
        sum = 0
        for population_idx in range(len(populations)):
            sum += _get_kl(population_idx, resolution)
        return sum / len(populations)

    for resolution in [1,2,3,5,10,15,25]:     
        print(_get_avg_kl(resolution))


get_kl()
