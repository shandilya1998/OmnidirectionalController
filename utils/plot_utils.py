import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import pandas as pd
import os

np.seterr('raise')

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    fig.savefig(os.path.join(log_folder, 'learning_curve.png'))

def plot_stability_metric(logdir, direction = 'forward'):
    info = pd.read_csv(os.path.join(logdir, 'info.csv'), index_col = 0)
    cond1 = (info['direction'] == 'forward') & (info['task'] == 'straight') & ((info['gait'] == 'trot') | (info['gait'] == 'ds_crawl'))
    cond2 = (((info['task'] == 'rotate') | (info['task'] == 'turn')) & (info['direction'] == 'left') & ((info['gait'] == 'trot') | (info['gait'] == 'ds_crawl')))
    info = info[cond1 | cond2].reset_index()
    items = ['d1', 'd2', 'd3', 'stability']
    items = {
        key : [
            np.load(
                os.path.join(
                    logdir,
                    '{}_{}.npy'.format(f, key)
                )
            ) for f in info.loc[:, 'id'].values.tolist()
        ] for key in items
    }

    length = info.loc[:, 'length'].min()
    step = int(length * 0.2)
    num_gaits = len(info)
    colors = [[np.random.random() for i in range(3)] for j in range(num_gaits)]
    for key in items.keys():
        fig, ax = plt.subplots(1,1,figsize = (7.5,7.5))
        for i in range(num_gaits):
            out = []
            mean = np.mean(items[key][i])
            std = np.std(items[key][i])
            items[key][i][items[key][i] > 2 * std + mean] = mean + 2 * std
            items[key][i][items[key][i] < -2 * std + mean] = mean - 2 * std
            for j in range(step, length):
                try:
                    out.append(np.mean(items[key][i][j-step:j]))
                except:
                    print(items[key][i][j - step: j])
                    print(np.sum(items[key][i][j - step: j]))
            ax.plot(np.array(out, dtype = np.float32), color = colors[i], label = '{}, {}, {}'.format(info.loc[i, 'gait'], info.loc[i, 'task'], info.loc[i, 'direction']))
        ax.set_xlabel('time', fontsize = 16)
        ax.set_ylabel(key, fontsize = 16)
        ax.legend()
        path = os.path.join(logdir, '{}.png'.format(key))
        plt.show()
        fig.savefig(path)
        plt.close()


def plot_rl_results(logdir, datadir):
    files = os.listdir(datadir)
    to_scale = ['TD3_38', 'TD3_7', 'PPO_25']
    dfs = [pd.read_csv(os.path.join(datadir, f)) for f in files]
    rewards = [df[df['metric'] == 'rollout/ep_rew_mean'].loc[:,'value'].values.tolist() for df in dfs]
    steps = [df[df['metric'] == 'rollout/ep_rew_mean'].loc[:, 'step'].values.tolist() for df in dfs]
    fig, ax = plt.subplots(1, 1, figsize = (10,6))
    length = int(1e6)
    colors = []
    indices = []
    for i, step in enumerate(steps):
        count = 0
        while step[count] < length:
            count += 1
        indices.append(count - 1)
        colors.append([np.random.random() for i in range(3)])
        max_val = max(rewards[i])
        if max_val > 500:
            rewards[i] = np.array(rewards[i]) * 500 / max_val
            rewards[i] = rewards[i].tolist()
    for i, (reward, step, f) in enumerate(zip(rewards, steps, files)):
        ax.plot(step[:indices[i]], reward[:indices[i]], color = colors[i], label = f[:-4])
    ax.set_xlabel('steps')
    ax.set_ylabel('reward')
    ax.legend()
    fig.savefig(os.path.join(logdir, 'reward.png'))
    plt.show()


