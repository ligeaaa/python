import numpy as np
import math
from scipy.stats import gamma

def generate_possion_process(T, lambda_):
    spike_train = np.random.poisson(lambda_, T)
    spike_times = np.cumsum(spike_train)
    return spike_times


def generate_gamma_process(shape, scale, size):
    spike_train = gamma.rvs(a=shape, scale=scale, size=size)
    spike_times = np.cumsum(spike_train)
    return spike_times


if __name__ == '__main__':
    T = 10000
    lambda_ = 10
    dt = 0.001

    spike_train = generate_possion_process(T, lambda_)


    # print(spike_train)