import numpy as np
import os
import matplotlib.pyplot as plt
from constants import params
import argparse
import shutil
from tqdm import tqdm

def hopf_simple_step(omega, mu, z, dt = 0.001):
    x, y = np.split(z, 2, -1)
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x) + dt * np.abs(omega) * 2
    r += dt * r * (mu - r ** 2)
    z = np.concatenate([
        r * np.cos(phi),
        r * np.sin(phi)
    ], -1)
    return z

def hopf(omega, mu, z, N, dt):
    Z = []
    for i in range(N):
        z = hopf_simple_step(omega, mu, z, dt)
        Z.append(z.copy())
    return np.stack(Z, 0)


def _get_pattern(thresholds, dx = 0.001):
    out = []
    x = 0.0 
    y = [0.9, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]
    while x < thresholds[1]:
        out.append(((y[1] - y[0])/(thresholds[1] - thresholds[0])) * (x - thresholds[0]) + y[0])
        x += dx
    while x < thresholds[2]:
        out.append(y[2])
        x += dx
    while x < thresholds[3]:
        out.append(((y[3] - y[2])/(thresholds[3] - thresholds[2])) * (x - thresholds[2]) + y[2])
        x += dx
    while x < thresholds[4]:
        out.append(y[4])
        x += dx
    while x < thresholds[5]:
        out.append(((y[5] - y[4])/(thresholds[5] - thresholds[4])) * (x - thresholds[4]) + y[4])
        x += dx
    while x < thresholds[6]:
        out.append(y[6])
        x += dx
    out = np.array(out, dtype = np.float32)
    return out 

def _get_polynomial_coef(degree, thresholds, dt = 0.001):
    y = _get_pattern(thresholds, dt) 
    x = np.arange(0, thresholds[-1], dt, dtype = np.float32)
    C = np.polyfit(x, y, degree)
    return C

def _plot_beta_polynomial(logdir, C, degree, thresholds, dt = 0.001):
    y = _get_pattern(thresholds, dt) 
    x = np.arange(0, thresholds[-1], dt, dtype = np.float32)
    def f(x, degree):
        X = np.array([x ** pow for pow in range(degree, -1, -1 )], dtype = np.float32)
        return np.sum(C * X)
    y_pred = np.array([f(x_, degree) for x_ in x], dtype = np.float32)
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    ax.plot(x, y, color = 'r', linestyle = ':', label = 'desired beta')
    ax.plot(x, y_pred, color = 'b', linestyle = '--', label = 'actual beta')
    ax.set_xlabel('omega')
    ax.set_ylabel('beta')
    ax.legend()
    fig.savefig(os.path.join(logdir, 'polynomial.png'))
    print(os.path.join(logdir, 'polynomial.png'))
    plt.close()
    print('Plot Finished')

def _get_beta(x, C, degree):
    x = np.abs(x)
    X = np.stack([x ** p for p in range(degree, -1, -1 )], 0)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)

def _get_omega_choice(phi):
    return np.tanh(1e3 * (phi))

def hopf_mod(omega, mu, z, C, degree, N, dt):
    Z = []
    for i in range(N):
        z, w = hopf_mod_step(omega, mu, z, C, degree, dt)
        Z.append(z.copy())
    return np.stack(Z, 0)


def hopf_mod_step(omega, mu, z, C, degree, dt = 0.001):
    """ 
        corresponding driving simple oscillator must have double the frequency
    """
    units_osc = z.shape[-1] // 2
    x, y = np.split(z, 2, -1) 
    r = np.sqrt(np.square(x) + np.square(y))
    beta = _get_beta(omega, C, degree)
    phi = np.arctan2(y, x)
    mean = np.abs(1 / (2 * beta * (1 - beta)))
    amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
    w = np.abs(omega) * (mean + amplitude * _get_omega_choice(phi))
    phi += dt * w 
    r += dt * (mu - r ** 2) * r 
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z_ = np.concatenate([x, y], -1) 
    return z_, w

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

def cpg_step(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    z1 = hopf_simple_step(omega, mu, z1, dt) 
    z2, w = hopf_mod_step(omega, mu, z2, C, params['degree'], dt) 
    x1, y1 = np.split(z1, 2, -1) 
    xs = np.cos(phase)
    ys = np.sin(phase)
    coupling = np.concatenate([
        xs * x1 - ys * y1, 
        xs * y1 + x1 * ys
    ], -1) 
    z2 += dt * params['coupling_strength'] * coupling
    return z2, w, z1

def cpg_step_v2(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    z1 = hopf_simple_step(omega, mu, z1, dt) 
    z2, w = hopf_mod_step(omega, mu, z2, C, params['degree'], dt) 
    x1, y1 = np.split(z1, 2, -1)
    p1 = np.arctan2(y1, x1)
    r1 = np.sqrt(x1 ** 2 + y1 ** 2)
    x1 = (
        r1 ** (np.abs(w) / np.abs(omega * 2))
    ) * np.cos(p1 * np.abs(w) / np.abs(omega * 2))
    y1 = (
        r1 ** (np.abs(w) / np.abs(omega * 2))
    ) * np.sin(p1 * np.abs(w) / np.abs(omega * 2))
    xs = np.cos(phase / np.abs(2 * omega))
    ys = np.sin(phase / np.abs(2 * omega))
    coupling = np.concatenate([
        xs * x1 - ys * y1, 
        xs * y1 + x1 * ys
    ], -1) 
    z2 += dt * params['coupling_strength'] * coupling
    return z2, w, z1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--num_osc',
        type = int,
        help = 'number of oscillators'
    )
    parser.add_argument(
        '--timesteps',
        type = int,
        default = 10000,
        help = 'number of timesteps to run oscillators for'
    )
    parser.add_argument(
        '--dt',
        type = float,
        default = 0.001,
        help = 'sampling period'
    )
    args = parser.parse_args()
    num_osc = args.num_osc
    N = args.timesteps
    dt = args.dt
    z = np.concatenate([np.zeros((num_osc,), dtype = np.float32), np.ones((num_osc,), dtype = np.float32)], -1)
    omega = np.arange(1, num_osc + 1, dtype = np.float32) * np.pi * 2 / (num_osc + 1)
    mu = np.ones((num_osc,), dtype = np.float32)
    print('Running Oscillators.')
    plot_path = os.path.join(args.out_path, 'plots')
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.mkdir(plot_path)
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50)
    np.save(open(os.path.join(plot_path, 'coef.npy'), 'wb'), C)
    _plot_beta_polynomial(plot_path, C, params['degree'], params['thresholds'], dt * 50)
    Z_hopf = hopf(omega.copy(), mu.copy(), z.copy(), N, dt)
    Z_mod = hopf_mod(omega.copy(), mu.copy(), z.copy(), C, params['degree'], N, dt)
    os.mkdir(os.path.join(plot_path, 'hopf'))
    T = np.arange(N, dtype = np.float32) * dt
    print('Plotting Output.')
    for i in tqdm(range(num_osc)):
        num_steps = int(num_osc / ((i + 1) * dt))
        fig, axes = plt.subplots(2,2, figsize = (12,12))
        axes[0][0].plot(T[:num_steps], Z_hopf[:num_steps, i], linestyle = ':', color = 'r', label = 'constant omega')
        axes[0][0].plot(T[:num_steps], Z_mod[:num_steps, i], color = 'b', label = 'variable omega')
        axes[0][0].set_xlabel('time (s)',fontsize=15)
        axes[0][0].set_ylabel('real part',fontsize=15)
        axes[0][0].set_title('Trend in Real Part',fontsize=15)
        axes[0][0].legend()
        axes[0][1].plot(T[:num_steps], Z_hopf[:num_steps, i + num_osc], linestyle = ':', color = 'r', label = 'constant omega')
        axes[0][1].plot(T[:num_steps], Z_mod[:num_steps, i + num_osc], color = 'b', label = 'variable omega')
        axes[0][1].set_xlabel('time (s)',fontsize=15)
        axes[0][1].set_ylabel('imaginary part',fontsize=15)
        axes[0][1].set_title('Trend in Imaginary Part',fontsize=15)
        axes[0][1].legend()
        axes[1][0].plot(Z_hopf[:, i], Z_hopf[:, i + num_osc], linestyle = ':', color = 'r', label = 'constant omega')
        axes[1][0].plot(Z_mod[:, i], Z_mod[:, i + num_osc], color = 'b', label = 'variable omega')
        axes[1][0].set_xlabel('real part',fontsize=15)
        axes[1][0].set_ylabel('imaginary part',fontsize=15)
        axes[1][0].set_title('Phase Space',fontsize=15)
        axes[1][0].legend()
        axes[1][1].plot(T[:num_steps], np.arctan2(Z_hopf[:num_steps, i], Z_hopf[:num_steps, i + num_osc]), linestyle = ':', color = 'r', label = 'constant omega')
        axes[1][1].plot(T[:num_steps], np.arctan2(Z_mod[:num_steps, i], Z_mod[:num_steps, i + num_osc]), color = 'b', label = 'variable omega')
        axes[1][1].set_xlabel('time (s)',fontsize=15)
        axes[1][1].set_ylabel('phase (radians)',fontsize=15)
        axes[1][1].set_title('Trend in Phase',fontsize=15)
        axes[1][1].legend()
        fig.savefig(os.path.join(plot_path, 'hopf', 'oscillator_{}.png'.format(i)))
        plt.close('all')
    phi = np.array([0.0, 0.25, 0.5, 0.75], dtype = np.float32)
    phi = phi + np.cos(phi * 2 * np.pi) * 3 * (1 - 0.75) / 8
    z = np.concatenate([np.cos(phi * 2 * np.pi), np.sin(phi * 2 * np.pi)], -1)
    omega = 1.6 * np.ones((4,), dtype = np.float32)
    mu = np.ones((4,), dtype = np.float32)
    Z_mod = hopf_mod(omega.copy(), mu.copy(), z.copy(), C, params['degree'], N, dt)
    fig, axes = plt.subplots(2,2, figsize = (10,10))
    num_osc = 4
    color = ['r', 'b', 'g', 'y']
    label = ['Phase {:2f}'.format(i) for i in [0.0, 0.25, 0.5, 0.75]]
    num_steps = int(2 * np.pi / (1.6 * dt))
    for i in tqdm(range(num_osc)):
        axes[0][0].plot(T[:num_steps], Z_mod[:num_steps, i], color = color[i], linestyle = '--')
        axes[0][0].set_xlabel('time (s)')
        axes[0][0].set_ylabel('real part')
        axes[0][0].set_title('Trend in Real Part')
        axes[0][1].plot(T[:num_steps], -np.maximum(-Z_mod[:num_steps, i + num_osc], 0), color = color[i], linestyle = '--')
        axes[0][1].set_xlabel('time (s)')
        axes[0][1].set_ylabel('imaginary part')
        axes[0][1].set_title('Trend in Imaginary Part')
        axes[1][0].plot(Z_mod[:, i], Z_mod[:, i + num_osc], color = color[i], linestyle = '--')
        axes[1][0].set_xlabel('real part')
        axes[1][0].set_ylabel('imaginary part')
        axes[1][0].set_title('Phase Space')
        axes[1][1].plot(T, np.arctan2(Z_mod[:, i], Z_mod[:, i + num_osc]), color = color[i], linestyle = '--')
        axes[1][1].set_xlabel('time (s)')
        axes[1][1].set_ylabel('phase (radian)')
        axes[1][1].set_title('Trend in Phase')
    fig.savefig(os.path.join(plot_path, 'phase_comparison.png'))
    plt.show()
    print('Done.')
    print('Thank You.')
