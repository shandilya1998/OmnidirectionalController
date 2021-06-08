import mujoco_py
import math
import matplotlib.pyplot as plt
from constants import params
import  numpy as np
import argparse
from constants import params
import os

def test_env_from_xml(path = 'assets/ant.xml'):
    f = open(path, 'r')
    model_xml = f.read()
    model = mujoco_py.load_model_from_xml(model_xml)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    t = 0
    while True:
        sim.data.ctrl[1] = math.sin(t / 20)
        sim.data.ctrl[0] = math.sin(t / 20)
        sim.data.ctrl[2] = math.sin(t / 20)
        sim.step()
        viewer.render()
        t += 1
        if t > 2000:
            break

def test_env(env):
    t = 0
    ac = env.action_space.sample()
    while True:
        ob = env.step(ac)
        env.render()
        t += 1
        if t > 100:
            break
    env.reset()
    fig, axes = plt.subplots(4,3, figsize = (15, 20))
    i = 0
    joint_pos = np.nan_to_num(np.vstack(env._track_item['joint_pos']))
    true_joint_pos = np.nan_to_num(np.vstack(env._track_item['true_joint_pos']))
    num_joints = joint_pos.shape[-1]
    t = np.arange(joint_pos.shape[0], dtype = np.float32) * env.dt
    while True:
        if i >= num_joints:
            break
        axes[int(i / 3)][i % 3].plot(t[:500], joint_pos[:500, i], color = 'r')
        axes[int(i / 3)][i % 3].plot(t[:500], true_joint_pos[:500, i], color = 'b')
        axes[int(i / 3)][i % 3].set_title('Joint {}'.format(i))
        axes[int(i / 3)][i % 3].set_xlabel('time (s)')
        axes[int(i / 3)][i % 3].set_ylabel('joint position (radian)')
        i += 1
    fig.savefig('assets/ant_joint_pos.png')

    return env

def test():
    omega = 6
    beta = 0.75
    omega_sw = 6 / (1 - beta)
    omega_st = 6 / beta
    dt = 0.001
    out = []
    out2 = []
    counter = 0
    phase = 0.0
    fig, ax = plt.subplots(1, 1, figsize = (5, 10))
    while counter < 3:
        while phase < np.pi / 2:
            out.append(np.sin(phase + np.pi))
            out2.append(0.0)
            phase += omega_st * dt
        while phase < 3 * np.pi / 2:
            out.append(np.sin(phase + np.pi))
            out2.append(np.cos(phase))
            phase += omega_sw * dt
        while phase <= 2 * np.pi:
            out.append(np.sin(phase + np.pi))
            out2.append(0.0)
            phase += omega_st * dt
        phase = phase % (2 * np.pi)
        counter += 1
    ax.plot(out)
    ax.plot(out2)
    plt.show()

def plot_tracked_item(name = 'position'):
    item = np.load(os.path.join('assets', 'ant_{}.npy'.format(name)))
    t = np.arange(item.shape[0], dtype = np.float32) * params['dt']
    fig, axes = plt.subplots(1,1, figsize = (5, 5))
    axes.plot(item[:, 0], label = 'x', color = 'b')
    axes.plot(item[:, 1], label = 'y', color = 'g')
    axes.plot(item[:, 2], label = 'z', color = 'c')
    axes.set_title('Position (m) vs Time (s)')
    axes.legend()
    axes.set_xlabel('time (s)')
    axes.set_ylabel('position (m)')
    fig.savefig(os.path.join('assets', 'ant_{}.png'.format(name)))

if __name__ =='__main__':
    from simulations.quadruped import Quadruped
    env = Quadruped('ant.xml')
    env = test_env(env)
    plot_tracked_item()
