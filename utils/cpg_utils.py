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
    lambdas = np.arange(1.0, 101, 50.0).tolist()
    fig, ax = plt.subplots(2, 2, figsize = (20, 20))
    steps = int(2 * np.pi / (2 * omega[0] * dt))
    z = mu * np.concatenate([np.cos(phase), np.sin(phase)], -1)
    Z, W = hopf_mod(omega, mu, z, C, params['degree'], N, dt)
    x, y = np.split(Z, 2, -1)
    phi_ref = np.arctan2(y, x)[:-(Z.shape[0] % steps)]
    def func(x):
        if x < 0:
            return x + 2 * np.pi
        else:
            return x
    func = np.vectorize(func)
    print(phi_ref.shape)
    phi_ref = func(phi_ref)
    r_ref = np.sqrt(np.square(x) + np.square(y))

    T = np.arange(N) * dt
    ax[1][0].plot(
        T[-steps:], Z[-steps:, 0],
        color = 'r',
        label = 'reference'
    )
    ax[1][1].plot(
        T[-steps:], Z[-steps::, 1],
        color = 'r',
        label = 'reference'
    )
    for lmbda in tqdm(lambdas):
        params['lambda'] = lmbda
        Z2, W2, Z1 = cpg(omega, mu, phase, C, params['degree'], N, dt)
        length = int(Z2.shape[0] - Z2.shape[0] % steps)
        x, y = np.split(Z2, 2, -1)
        r = np.linalg.norm(
            np.sqrt(np.square(x) + np.square(y)) - r_ref,
            axis = -1
        )
        phi = np.arctan2(y, x)[:length, :]
        err = np.sqrt(np.square(
            np.gradient(func(phi), axis = 0) - np.gradient(phi_ref, axis = 0),
        ))
        err2 = np.sqrt(np.square(W2  - W))[:length, :]
        err = err.reshape(int(err.shape[0] / steps), steps, err.shape[-1])
        err = np.sum(err, 1)
        err2 = err2.reshape(int(err2.shape[0] / steps), steps, err2.shape[-1])
        err2 = np.sum(err2, 1)
        color = np.random.uniform(low = 0.0, high = 1.0, size = (3,))
        ax[0][0].plot(T, r,
            color = color,
            label = '\u03BB: {}'.format(lmbda))
        ax[0][1].plot(np.arange(err.shape[0]), err,
            color = color,
            label = '\u03BB: {}'.format(lmbda))
        ax[0][1].plot(np.arange(err2.shape[0]), err2,
            color = color,
            label = '\u03BB: {}'.format(lmbda))
        ax[1][0].plot(
            T[-steps:], Z2[-steps:, 0],
            color = color,
            label = '\u03BB: {}'.format(lmbda)
        )
        ax[1][1].plot(
            T[-steps:], Z2[-steps:, 1],
            color = color,
            label = ' \u03BB: {}'.format(lmbda)
        )
    ax[0][0].set_xlabel('time')
    ax[0][0].set_ylabel('r')
    ax[0][1].set_xlabel('no. of periods')
    ax[0][1].set_ylabel('err in phase velocity')
    ax[1][0].set_xlabel('time')
    ax[1][0].set_ylabel('real part')
    ax[1][1].set_xlabel('time')
    ax[1][1].set_ylabel('imaginary part')
    ax[0][0].legend(loc = 'upper right')
    ax[0][1].legend(loc = 'upper right')
    ax[1][0].legend(loc = 'upper right')
    ax[1][1].legend(loc = 'upper right')
    plt.show()
