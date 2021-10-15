import numpy as np
import matplotlib.pyplot as plt
from nwlib import ModHopf_rescale, _get_omega_choice

def hopf_simple_step(omega, mu, z, dt = 0.005):
    x, y = np.split(z, 2, -1)
    r = np.sqrt(x * x + y * y)
    x = x + ((mu - r * r) * x - omega * y) * dt
    y = y + ((mu - r * r) * y + omega * x) * dt
    z = np.concatenate([x, y], -1)
    return z

if __name__ == '__main__':
    omega = np.array([[5.78]])
    mu = np.ones((1, 1))
    dt = 0.005
    N = int(1e5)
    lst_ref = []
    z = np.array([[1], [0]])
    z_ref = z.copy()
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    model = ModHopf_rescale(omega, 1.0)
    lst = []
    for i in range(N):
        x, y = np.split(z_ref, 2, 0)
        phi = np.arctan2(y ,x)
        w = omega * (
            model.mean + model.amplitude * _get_omega_choice(phi)
        ) / 2
        z_ref = hopf_simple_step(w, mu, z_ref, dt)
        z = z + model.dif(z) * dt
        lst_ref.append(z_ref.copy())
        lst.append(z.copy())
        axes[0].plot(
            np.stack(lst, 0)[:, 0, 0],
            color = 'r',
            linestyle = '--',
            label = 'test')
        axes[0].plot(
            np.stack(lst_ref, 0)[:, 0, 0],
            color = 'b',
            linestyle = '--',
            label = 'reference')
        axes[1].plot(
            np.stack(lst, 0)[:, 1, 0], 
            color = 'r',
            linestyle = '--',
            label = 'test')
        axes[1].plot(
            np.stack(lst_ref, 0)[:, 1, 0], 
            color = 'b',
            linestyle = '--',
            label = 'reference')
        plt.pause(0.005)
