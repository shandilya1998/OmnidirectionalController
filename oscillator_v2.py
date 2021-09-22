import os
import numpy as np
import matplotlib.pyplot as plt
from constants import params

def hopf_simple_step(omega, mu, z, dt = 0.001):
    x, y = np.split(z, 2, -1)
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x) + dt * omega * 4
    r += dt * r * (mu - r ** 2)
    z = np.concatenate([
        r * np.cos(phi),
        r * np.sin(phi)
    ], -1)
    return z

def test_hopf_simple(
        logdir = 'assets/out/plots',
        filename = 'test_hopf_simple.png'
    ):
    omega = np.array([0.2, 0.4, 0.6, 0.8]) * 2 * np.pi
    mu = np.ones((4,))
    z = np.concatenate([
        np.ones((4,)),
        np.zeros((4,))
    ])
    N = 10000
    dt = 0.001
    T = np.arange(N) * dt
    Z = [z.copy()]
    for i in range(N):
        z = hopf_simple_step(omega, mu, z, dt)
        Z.append(z.copy())
    Z = np.stack(Z, 0)
    fig, axes = plt.subplots(4,2,figsize=(12, 24))
    for i in range(4):
        axes[i][0].plot(T[-N // 2:], Z[-N // 2:, i])
        axes[i][0].set_xlabel('time')
        axes[i][0].set_ylabel('real part')
        axes[i][1].plot(T[-N // 2:], Z[-N // 2:, i + 4])
        axes[i][1].set_xlabel('time')
        axes[i][1].set_ylabel('imaginary part')
    fig.savefig(os.path.join(logdir, filename))
    plt.show()
    plt.close('all')

def _get_beta(x, C, degree):
    x = np.abs(x)
    X = np.stack([x ** p for p in range(degree, -1, -1 )], 0)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)

def _get_omega_choice(phi):
    return np.tanh(1e3 * (phi))

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
    w = np.abs(omega) * (mean + amplitude * _get_omega_choice(phi)) * 2
    phi += dt * w
    r += dt * (mu - r ** 2) * r
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z_ = np.concatenate([x, y], -1)
    return z_, w

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

def test_hopf_mod(
        logdir = 'assets/out/plots',
        filename = 'test_hopf_mod.png'
    ):
    omega = np.array([0.2, 0.4, 0.6, 0.8]) * 2 * np.pi
    mu = np.ones((4,))
    z = np.concatenate([
        np.ones((4,)),
        np.zeros((4,))
    ])
    dt = 0.001
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50)
    N = 10000
    T = np.arange(N) * dt
    Z = [z.copy()]
    for i in range(N):
        z, w = hopf_mod_step(omega, mu, z, C, params['degree'], dt)
        Z.append(z.copy())
    Z = np.stack(Z, 0)
    fig, axes = plt.subplots(4,2,figsize=(12, 24))
    for i in range(4):
        axes[i][0].plot(T[-N // 2:], Z[-N // 2:, i])
        axes[i][0].set_xlabel('time')
        axes[i][0].set_ylabel('real part')
        axes[i][1].plot(T[-N // 2:], Z[-N // 2:, i + 4])
        axes[i][1].set_xlabel('time')
        axes[i][1].set_ylabel('imaginary part')
    fig.savefig(os.path.join(logdir, filename))
    plt.show()
    plt.close('all')


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

def cpg_step_v1(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    z1 = hopf_simple_step(omega, mu, z1, dt)
    z2, w = hopf_mod_step(omega, mu, z2, C, params['degree'], dt)
    x1, y1 = np.split(z1, 2, -1)
    x2, y2 = np.split(z2, 2, -1)
    zs = mu * np.exp(1j * phase)
    z2 = x2 + 1j * y2
    z1 = x1 + 1j * y1
    z2 += dt * params['coupling_strength'] * zs * z1
    z2 = np.concatenate([
        np.real(z2),
        np.imag(z2)
    ], -1)
    z1 = np.concatenate([
        np.real(z1),
        np.imag(z1)
    ], -1)
    return z2, w, z1


def cpg_step_v2(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    x1, y1 = np.split(z1, 2, -1)
    phase_ = np.arctan2(y1, x1) + omega * dt
    r = np.sqrt(np.square(x1) + np.square(y1))
    xs = r * np.cos(phase_ + phase)
    ys = r * np.sin(phase_ + phase)
    zs = np.concatenate([
        xs, ys
    ], -1)
    x1 = r * np.cos(phase_)
    y1 = r * np.sin(phase_)
    z1 = np.concatenate([x1, y1], -1)
    z2, w = hopf_mod_step(omega, mu, z2, C, params['degree'], dt) 
    z2 += dt * params['coupling_strength'] * zs
    return z2, w, z1

def cpg_step_v3(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    x1, y1 = np.split(z1, 2, -1) 
    phase_ = np.arctan2(y1, x1) + omega * dt + phase
    r = np.sqrt(np.square(x1) + np.square(y1))
    r += dt * r * (mu - r ** 2)
    x1 = np.cos(phase_)
    y1 = np.sin(phase_)
    z1 = np.concatenate([x1, y1], -1)
    z2, w = hopf_mod_step(omega, mu, z2, C, params['degree'], dt)
    z2 += dt * params['coupling_strength'] * z1
    return z2, w, z1

def cpg_step_v4(omega, mu, z1, z2, phase, C, degree, dt = 0.001):
    x1, y1 = np.split(z1, 2, -1)
    phase_ = np.arctan2(y1, x1) + omega * dt * 4
    r = np.sqrt(np.square(x1) + np.square(y1))
    r += dt * r * (mu - r ** 2)
    x1 = r * np.cos(phase_)
    y1 = r * np.sin(phase_)
    z1 = np.concatenate([x1, y1], -1)
    units_osc = z2.shape[-1] // 2
    x2, y2 = np.split(z2, 2, -1)
    r = np.sqrt(np.square(x2) + np.square(y2))
    beta = _get_beta(omega, C, degree)
    phi = np.arctan2(y2, x2)
    phi = (phi + np.pi - phase) % (2 * np.pi) - np.pi
    mean = np.abs(1 / (2 * beta * (1 - beta)))
    amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
    w = np.abs(omega) * (mean + amplitude * _get_omega_choice(phi)) * 2
    phi += dt * w
    r += dt * (mu - r ** 2) * r
    phi = (phi + np.pi + phase) % (2 * np.pi) - np.pi
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z2 = np.concatenate([x, y], -1)
    return z2, w, z1

def test_cpg(
        show = False,
        version = 0,
        logdir = 'assets/out/plots',
        filename = 'test_cpg',
        extension = 'png',
        size = 15,
    ):
    func = cpg_step
    if version == 0:
        pass
    elif version == 1:
        func = cpg_step_v1
    elif version == 2:
        func = cpg_step_v2
    elif version == 3:
        func = cpg_step_v3
    elif version == 4:
        func = cpg_step_v4
    phase = np.array([0.0, 0.25, 0.5, 0.75] * 4) * 2 * np.pi
    omega = np.array([0.2, 0.2, 0.2, 0.2]) * 2 * np.pi
    omega = np.concatenate([omega, 2 * omega, 3 * omega, 4 * omega], -1)
    mu = np.ones((4 * 4,))
    dt = 0.01
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50) 
    N = 1000
    T = np.arange(N) * dt
    z2 = np.concatenate([
        np.ones((4 * 4,)),
        np.zeros((4 * 4,))
    ])
    z1 = z2.copy()
    Z2 = [z2.copy()]
    Z1 = [z1.copy()]
    for i in range(N):
        z2, _, z1 = func(omega, mu, z1, z2, phase, C, params['degree'], dt)
        Z2.append(z2.copy())
        Z1.append(z1.copy())
    Z1 = np.stack(Z1, 0)
    Z2 = np.stack(Z2, 0)
    plt.rcParams["font.size"] = "12"
    for j in range(4):
        fig, axes = plt.subplots(4,3,figsize=(3 * size, 4 * size))
        steps = N // 8
        for i in range(4):
            phase_1 = (1.0 + np.arctan2(Z1[-steps:, i + j * 4], Z1[-steps:, i + 4 * 4 + j * 4]) / np.pi) / 2
            phase_2 = (1.0 + np.arctan2(Z2[-steps:, i + j * 4], Z2[-steps:, i + 4 * 4 + j * 4]) / np.pi) / 2
            diff = phase_2 - phase_1
            axes[i][0].plot(T[-steps:], Z1[-steps:, i + j * 4], '--b', label = 'reference') 
            axes[i][1].plot(T[-steps:], Z1[-steps:, i + 4 * 4 + j * 4], '--b', label = 'reference') 
            axes[i][2].plot(
                T[-steps:],
                phase_1,
                '--b',
                label = 'reference'
            )
            axes[i][0].plot(T[-steps:], Z2[-steps:, i + j * 4], '--r', label = 'generator')
            axes[i][1].plot(T[-steps:], Z2[-steps:, i + 4 * 4 + j * 4], '--r', label = 'generator')
            axes[i][2].plot(
                T[-steps:],
                phase_2,
                '--r',
                label = 'generator'
            )
            axes[i][2].plot(
                T[-steps:],
                diff,
                '--g',
                label = 'phase difference'
            )
            axes[i][0].set_xlabel('time')
            axes[i][0].set_ylabel('real part')
            axes[i][1].set_xlabel('time')
            axes[i][1].set_ylabel('imaginary part')
            axes[i][2].set_xlabel('time')
            axes[i][2].set_ylabel('phase')
            axes[i][0].legend(loc = 'upper left')
            axes[i][1].legend(loc = 'upper left')
            axes[i][2].legend(loc = 'upper left')
        name = '{}_{}_{}.{}'.format(
            filename,
            str(version),
            str(j),
            extension
        )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels[:1])]
        fig.legend(lines, labels)
        fig.savefig(os.path.join(logdir, name))
        if show:
            plt.show()
        plt.close()

def test_driven_cpg(
        version = 0,
        logdir = 'assets/out/plots',
        filename = 'test_driven_cpg',
        extension = 'png',
    ):
    func = cpg_step_v3
    if version == 0:
        pass
    phase = np.array([0.0, 0.25, 0.5, 0.75]) * 2 * np.pi
    omega = np.array([0.25, 0.25, 0.25, 0.25]) * 2 * np.pi
    mu = np.ones((4,))
    dt = 0.001
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50)
    N = 10000
    T = np.arange(N) * dt
    z2 = np.concatenate([
        np.ones((4,)),
        np.zeros((4,))
    ])
    z1 = z2.copy()
    Z2 = [z2.copy()]
    Z1 = [z1.copy()]
    for i in range(N):
        z2, _, z1 = func(omega, mu, z1, z2, phase, C, params['degree'], dt)
        Z2.append(z2.copy())
        Z1.append(z1.copy())
    Z1 = np.stack(Z1, 0)
    Z2 = np.stack(Z2, 0)
    fig, axes = plt.subplots(4,3,figsize=(3 * 15, 4 * 15))
    steps = N // 8
    for i in range(4):
        axes[i][0].plot(T[-steps:], Z1[-steps:, i], '--b', label = 'reference')
        axes[i][1].plot(T[-steps:], Z1[-steps:, i + 4], '--b', label = 'reference')
        phase_1 = (1.0 + np.arctan2(Z1[:, i], Z1[:, i + 4]) / np.pi) / 2
        axes[i][2].plot(
            T[-steps:],
            phase_1[-steps:],
            '--b',
            label = 'reference'
        )
        axes[i][0].plot(T[-steps:], Z2[-steps:, i], '--r', label = 'generator')
        axes[i][1].plot(T[-steps:], Z2[-steps:, i + 4], '--r', label = 'generator')
        phase_2 = (1.0 + np.arctan2(Z2[:, i], Z2[:, i + 4]) / np.pi) / 2
        axes[i][2].plot(
            T[-steps:],
            phase_2[-steps:],
            '--r',
            label = 'generator'
        )
        diff = phase_2 - phase_1
        axes[i][2].plot(
            T[-steps:],
            diff[-steps:],
            '--g',
            label = 'phase difference'
        )
        axes[i][0].set_xlabel('time')
        axes[i][0].set_ylabel('real part')
        axes[i][1].set_xlabel('time')
        axes[i][1].set_ylabel('imaginary part')
        axes[i][2].set_xlabel('time')
        axes[i][2].set_ylabel('phase')
        axes[i][0].legend()
        axes[i][1].legend()
        axes[i][2].legend()
    filename = '{}_{}.{}'.format(
        filename,
        str(version),
        extension
    )
    fig.savefig(os.path.join(logdir, filename))
    plt.show()
    plt.close('all')

def hopf_mod_step_v2(omega, mu, z, B, C, degree, dt = 0.001):
    units_osc = z.shape[-1] // 2
    x, y = np.split(z, 2, -1)
    r = np.sqrt(np.square(x) + np.square(y))
    beta = _get_beta(omega, C, degree)
    phi = np.arctan2(y, x)
    mean = np.abs(1 / (2 * beta * (1 - beta)))
    amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
    w = np.abs(omega) * (mean + amplitude * _get_omega_choice(phi)) * 2 
    phi += dt * w 
    r += dt * (mu - B * (r ** 2)) * r 
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.concatenate([x, y], -1) 
    return z, w 

def cpg_step_v5(omega, mu, beta, z1, z2, phase, C, degree, dt = 0.001):
    units_osc = z1.shape[-1] // 2
    z1 = hopf_simple_step(omega, mu, z1, dt) 
    z2, w = hopf_mod_step_v2(
        omega, mu, z2, beta, C, params['degree'], dt
    )
    x1, y1 = np.split(z1, 2, -1)
    xs = np.cos(phase)
    ys = np.sin(phase)
    coupling = np.concatenate([
        xs * x1 - ys * y1, 
        xs * y1 + x1 * ys
    ], -1) 
    z2 += dt * params['coupling_strength'] * coupling
    return z2, w, z1

def test_cpg_v2(
        show = False,
        version = 5,
        logdir = 'assets/out/plots',
        filename = 'test_cpg_v2',
        extension = 'png',
        _phase = 0.75,
        num_osc = 10
    ):
    func = cpg_step_v5
    if version == 5:
        pass
    phase = np.array([_phase] * num_osc) * 2 * np.pi
    omega = np.array([0.2] * num_osc) * 2 * np.pi
    mu = np.ones((num_osc,)) * 0.81
    beta = 5 * np.arange(1, num_osc + 1) / num_osc
    dt = 0.001
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50)
    N = 10000
    T = np.arange(N) * dt
    z2 = np.concatenate([
        np.ones((num_osc,)),
        np.zeros((num_osc,))
    ])
    z2 = np.random.random((2 * num_osc,))
    z1 = np.concatenate([
        np.array([np.random.random()] * num_osc),
        np.array([np.random.random()] * num_osc)
    ], -1)
    Z2 = []
    Z1 = []
    PHI = []
    for i in range(N):
        z2, w, z1 = func(omega, mu, beta, z1, z2, phase, C, params['degree'], dt)
        Z2.append(z2.copy())
        Z1.append(z1.copy())
        x1, y1 = np.split(z1, 2, -1)
        x2, y2 = np.split(z2, 2, -1)
        phi = (np.arctan2(
                y2,
                x2
            ) - np.arctan2(
                y1, 
                x1
            )) / (2 * np.pi)
        PHI.append(phi.copy())
    Z1 = np.stack(Z1, 0)
    Z2 = np.stack(Z2, 0)
    PHI = np.stack(PHI, 0)
    plt.rcParams["font.size"] = "12"
    fig, axes = plt.subplots(num_osc, 4,figsize=(4 * 15, num_osc * 10)) 
    steps = N // 8
    for i in range(num_osc):
        axes[i][0].plot(T[-steps:], Z1[-steps:, i], '--b', label = 'reference')
        axes[i][1].plot(T[-steps:], Z1[-steps:, i + num_osc], '--b', label = 'reference')
        axes[i][2].plot(
            T[-steps:],
            (1.0 + np.arctan2(
                Z1[-steps:, i], Z1[-steps:, i + num_osc]
            ) / np.pi) / 2,
            '--b',
            label = 'reference'
        )
        axes[i][3].plot(
            T,
            np.repeat(np.expand_dims(phase, 0), N, 0)[:, i] / (2 * np.pi),
            '--b',
            label = 'reference'
        )
        axes[i][0].plot(T[-steps:], Z2[-steps:, i], '--r', label = 'generator')
        axes[i][1].plot(T[-steps:], Z2[-steps:, i + num_osc], '--r', label = 'generator')
        axes[i][2].plot(
            T[-steps:],
            (1.0 + np.arctan2(
                Z2[-steps:, i], Z2[-steps:, i + num_osc]
            ) / np.pi) / 2,
            '--r',
            label = 'generator'
        )
        axes[i][2].plot(
            T[-steps:],
            (
                np.arctan2(Z1[-steps:, i + num_osc], Z1[-steps:, i]) - \
                    np.arctan2(Z2[-steps:, i + num_osc], Z2[-steps:, i])
            ) / (2 * np.pi),
            '--g',
            label = 'phase difference'
        )
        axes[i][0].set_xlabel('time')
        axes[i][0].set_ylabel('real part')
        axes[i][1].set_xlabel('time')
        axes[i][1].set_ylabel('imaginary part')
        axes[i][2].set_xlabel('time')
        axes[i][2].set_ylabel('phase')
        axes[i][3].plot(
            T,  
            (np.arctan2(Z1[:, i + num_osc], Z1[:, i]) - \
                np.arctan2(Z2[:, i + num_osc], Z2[:, i])) / (2 * np.pi),
            '--b',
            label = 'generator'
        )
        axes[i][3].set_xlabel('time')
        axes[i][3].set_ylabel('phase difference')
        axes[i][0].legend(loc = 'upper left')
        axes[i][1].legend(loc = 'upper left')
        axes[i][2].legend(loc = 'upper left')
        axes[i][3].legend(loc = 'upper left')
    name = '{}_{}_{}.{}'.format(
        filename,
        str(version),
        str(i),
        extension
    )
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels[:1])]
    #fig.legend(lines, labels)
    fig.savefig(os.path.join(logdir, name))
    if show:
        plt.show()
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize = (21, 7))
    color = [np.random.random((3,)) for i in range(num_osc)]
    ax[0].plot(
        T,
        np.repeat(
            np.expand_dims(
                np.array(
                    [_phase] * num_osc
                ), 0),
            N,
            0
        ),
        color = 'r',
        linestyle = '--',
    )
    for i in range(num_osc):
        ax[0].plot(
            T,
            PHI[:, i],
            color = color[i],
            linestyle = '-',
            label = '\u03B2 =' + str(beta[i])
        )
        ax[1].plot(
            T,
            np.sqrt(np.square(Z1[:, i]) + np.square(Z1[:, i + num_osc])),
            color = color[i],
            linestyle = '-',
            label = '\u03B2 =' + str(beta[i])
        )
        ax[2].plot(
            T,
            np.sqrt(np.square(Z2[:, i]) + np.square(Z2[:, i + num_osc])),
            color = color[i],
            linestyle = '-',
            label = '\u03B2 =' + str(beta[i])
        )
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('phase difference')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('amplitude')
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('amplitude')
    name = 'test_cpg_phase_convergence_{}_{}.png'.format(
        version,
        str(int(_phase * 100))
    )
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels[:1])]
    fig.legend(lines, labels)
    fig.savefig(os.path.join(logdir, name))
    if show:
        plt.show()
    plt.close()
