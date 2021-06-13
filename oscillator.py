import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import shutil

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
        phi = phi + dt * omega
        r = r + dt * (mu - r ** 2) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        Z.append(z.copy())
    return np.stack(Z, 0)

def _get_beta(omega):
    return 0.625 - 2.210 / (omega + 2.326)

def hopf_mod(omega, mu, z, N = 10000, dt = 0.001):
    Z = []
    for i in tqdm(range(N)):
        units_osc = z.shape[-1]
        x, y = np.split(z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y,x)
        beta = _get_beta(omega)
        w = omega * (1 + np.sign(beta) * np.power(np.abs(beta), 5/12) * np.sin(phi))
        phi = phi + dt * w
        r = r + dt * (mu - r ** 2) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        Z.append(z.copy())
    return np.stack(Z, 0)

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
    z = np.concatenate([np.ones((num_osc,), dtype = np.float32), np.zeros((num_osc,), dtype = np.float32)], -1)
    omega = np.arange(1, num_osc + 1, dtype = np.float32) * np.pi * 2 / num_osc
    mu = np.ones((num_osc,), dtype = np.float32)
    print('Running Oscillators.')
    Z_hopf = hopf(omega.copy(), mu.copy(), z.copy(), N, dt)
    Z_mod = hopf_mod(omega.copy(), mu.copy(), z.copy(), N, dt)
    plot_path = os.path.join(args.out_path, 'plots')
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.mkdir(os.path.join(args.out_path, 'plots'))
    os.mkdir(os.path.join(args.out_path, 'plots', 'hopf'))
    T = np.arange(N, dtype = np.float32) * dt
    print('Plotting Output.')
    for i in tqdm(range(num_osc)):
        fig, axes = plt.subplots(2,2, figsize = (10,10))
        axes[0][0].plot(T, Z_hopf[:, i], linestyle = ':', color = 'r', label = 'constant omega')
        axes[0][0].plot(T, Z_mod[:, i], color = 'b', label = 'variable omega')
        axes[0][0].set_xlabel('time (s)')
        axes[0][0].set_ylabel('real part')
        axes[0][0].set_title('Trend in Real Part')
        axes[0][0].legend()
        axes[0][1].plot(T, Z_hopf[:, i + num_osc], linestyle = ':', color = 'r', label = 'constant omega')
        axes[0][1].plot(T, Z_mod[:, i + num_osc], color = 'b', label = 'variable omega')
        axes[0][1].set_xlabel('time (s)')
        axes[0][1].set_ylabel('imaginary part')
        axes[0][1].set_title('Trend in Imaginary Part')
        axes[0][1].legend()
        axes[1][0].plot(Z_hopf[:, i], Z_hopf[:, i + num_osc], linestyle = ':', color = 'r', label = 'constant omega')
        axes[1][0].plot(Z_mod[:, i], Z_mod[:, i + num_osc], color = 'b', label = 'variable omega')
        axes[1][0].set_xlabel('real part')
        axes[1][0].set_ylabel('imaginary part')
        axes[1][0].set_title('Phase Space')
        axes[1][0].legend()
        axes[1][1].plot(T, np.sqrt(np.square(Z_hopf[:, i]) + np.square(Z_hopf[:, i + num_osc])), linestyle = ':', color = 'r', label = 'constant omega')
        axes[1][1].plot(T, np.sqrt(np.square(Z_mod[:, i]) + np.square(Z_mod[:, i + num_osc])), color = 'b', label = 'variable omega')
        axes[1][1].set_xlabel('time (s)')
        axes[1][1].set_ylabel('magnitude')
        axes[1][1].set_title('Trend in Magnitude')
        axes[1][1].legend()
        fig.savefig(os.path.join(args.out_path, 'plots', 'hopf', 'oscillator_{}.png'.format(i)))
        plt.close('all')
    print('Done.')
    print('Thank You.')

