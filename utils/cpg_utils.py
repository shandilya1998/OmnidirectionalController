import numpy as np
import pandas as pd
from constants import params
import matplotlib.pyplot as plt
from tqdm import tqdm

def findLocalMaximaMinima(n, arr):

    # Empty lists to store points of
    # local maxima and minima
    mx = []
    mn = []

    # Checking whether the first point is
    # local maxima or minima or neither
    if(arr[0] > arr[1]):
        mx.append(0)
    elif(arr[0] < arr[1]):
        mn.append(0)

    # Iterating over all points to check
    # local maxima and local minima
    for i in range(1, n-1):

        # Condition for local minima
        if(arr[i-1] > arr[i] < arr[i + 1]):
            mn.append(i)

        # Condition for local maxima
        elif(arr[i-1] < arr[i] > arr[i + 1]):
            mx.append(i)

    # Checking whether the last point is
    # local maxima or minima or neither
    if(arr[-1] > arr[-2]):
        mx.append(n-1)
    elif(arr[-1] < arr[-2]):
        mn.append(n-1)

        # Print all the local maxima and
        # local minima indexes stored
    return np.array(mx), np.array(mn)

def test_cpg_entrainment(cpg, C, hopf_mod):
    omega = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (1,))
    mu = np.random.uniform(low = 0.0, high = 1.0, size = (1,))
    phase = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (1,))
    N = 10000
    dt = 0.005
    lambdas = np.arange(1.0, 100, 5.0).tolist()
    fig, ax = plt.subplots(1, 2, figsize = (20, 10))
    steps = int(2 * np.pi / (omega[0] * dt))
    params['dt'] = 0.005
    z = np.concatenate([np.cos(phase), np.sin(phase)], -1)
    Z = hopf_mod(omega, mu, z, C, params['degree'], N, dt)
    x, y = np.split(Z, 2, -1)
    phi_ref = np.arctan2(y, x)
    
    def func(x):
        if x < 0:
            return x + 2 * np.pi
        else:
            return x
    func = np.vectorize(func)
    phi_ref = func(phi_ref)

    for lmbda in tqdm(lambdas):
        params['lambda'] = lmbda
        Z2, W, Z1 = cpg(omega, mu, phase, C, params['degree'], N, dt)
        x, y = np.split(Z2, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.linalg.norm(func(np.arctan2(y, x)) - phi_ref, axis = -1)
        color = np.random.uniform(low = 0.0, high = 1.0, size = (3,))
        T = np.arange(N) * 5 *  params['dt']
        ax[0].plot(T, r,
            color = color,
            label = '\u03BB: {}'.format(lmbda))
        ax[1].plot(T, phi,
            color = color,
            label = '\u03BB: {}'.format(lmbda))
    ax[0].legend(loc = 'upper right')
    ax[1].legend(loc = 'upper right')
    plt.show()
