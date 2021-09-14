import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import shutil
from constants import params

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

def complex_multiply(z1, z2):
    x1, y1 = np.split(z1, 2, -1)
    x2, y2 = np.split(z2, 2, -1)
    return np.concatenate([
        x1 * x2 - y2 * y1,
        x1 * y2 + x2 * y1
    ], -1)


def hopf(omega, mu, z, N = 10000, dt = 0.001):
    Z = []
    for i in tqdm(range(N)):
        units_osc = z.shape[-1]
        x, y = np.split(z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y,x)
        phi = phi + dt * np.sqrt(np.square(omega))
        r = r + dt * (mu - r ** 2) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        Z.append(z.copy() * np.concatenate([np.tanh(1e3 * omega)] * 2, -1))
    return np.stack(Z, 0)

def plot_hopf_amplitude(logdir, omega, mu, z, N = 10000, dt = 0.001):
    Z = []
    for i in tqdm(range(N)):
        units_osc = z.shape[-1]
        x, y = np.split(z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y,x)
        phi = phi + dt * np.sqrt(np.square(omega))
        r = r + dt * (mu - r ** 2) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        Z.append(z.copy() * np.concatenate([np.tanh(1e3 * omega)] * 2, -1))

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    T = np.arange(N) * dt
    ax.plot(T, np.sqrt(np.sum(np.square(Z), -1)), color = 'b', linestyle = '--')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude')
    ax.set_title('Amplitude vs Time')
    fig.savefig(os.path.join(logdir, 'amplitude_hopf.png'))
    plt.show()
    plt.close()

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
    x = np.sqrt(np.square(x))
    X = np.array([x ** pow for pow in range(degree, -1, -1 )], dtype = np.float32)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)

def _get_omega_choice(phi):
    return np.tanh(1e3 * (phi))

def hopf_mod(omega, mu, z, C, degree, N = 10000, dt = 0.001):
    Z = []
    for i in tqdm(range(N)):
        units_osc = z.shape[-1]
        x, y = np.split(z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y,x)
        beta = _get_beta(omega, C, degree)
        mean = np.abs(1 / (2 * beta * (1 - beta)))
        amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
        w = np.sqrt(np.square(omega)) * (mean + amplitude * _get_omega_choice(phi)) / 2
        """
            [-pi, 0] - stance phase
            [0, pi] - swing phase
        """
        phi = phi + dt * w
        r = r + dt * (mu - r ** 2) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        Z.append(z.copy() * np.concatenate([np.tanh(1e3 * omega)] * 2, -1))
    return np.stack(Z, 0)

def hopf_step(omega, mu, z, C, degree, dt = 0.001):
    units_osc = z.shape[-1]
    x, y = np.split(z, 2, -1)
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y,x)
    beta = _get_beta(omega, C, degree)
    mean = np.abs(1 / (2 * beta * (1 - beta)))
    amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
    w = 2 * np.abs(omega) * (mean + amplitude * _get_omega_choice(phi))
    #print(w)
    """
        [-pi, 0] - stance phase
        [0, pi] - swing phase
    """
    phi = phi + dt * w
    r = r + dt * (mu - r ** 2) * r
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.concatenate([x, y], -1)
    return z, w

def _coupling(z, weights, units_osc):
    x1 = []
    for i in range(units_osc):
        indices = list(range(units_osc))
        d = indices.pop(i)
        x1.append(z[indices])
    x1 = np.stack(x1, axis = 0)
    out = np.multiply(np.repeat(np.expand_dims(z, 0), units_osc, 0), weights)
    out = np.sum(out, axis = -1)
    return out

def _coupled_hopf_step(omega, mu, z, weights, dt = 0.001):
    units_osc = z.shape[-1] // 2
    x, y = np.split(z, 2, -1) 
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y,x)
    w = 2 * np.abs(omega)
    #print(w)
    """ 
        [-pi, 0] - stance phase
        [0, pi] - swing phase
    """
    phi = phi + dt * w 
    r = r + dt * (mu - r ** 2) * r + _coupling(z, weights, units_osc)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.concatenate([x, y], -1) 
    return z, w


def _coupled_mod_hopf_step(omega, mu, z, C, degree, weights, dt = 0.001):
    units_osc = z.shape[-1] // 2
    x, y = np.split(z, 2, -1) 
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y,x)
    beta = _get_beta(omega, C, degree)
    mean = np.abs(1 / (2 * beta * (1 - beta)))
    amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
    w = 2 * np.abs(omega) * (mean + amplitude * _get_omega_choice(phi))
    #print(w)
    """ 
        [-pi, 0] - stance phase
        [0, pi] - swing phase
    """
    phi = phi + dt * w 
    r = r + dt * (mu - r ** 2) * r + _coupling(z, weights, units_osc)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.concatenate([x, y], -1) 
    return z, w

def test_coupled_mod_hopf(units_osc, omega, mu, z, C, degree, weights, dt = 0.001): 
    return True

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

