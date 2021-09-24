import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import pandas as pd
import os
import shutil
from constants import params
from oscillator_v2 import hopf_mod_step, _get_polynomial_coef, \
    hopf_simple_step


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

def _sort(x, y):
    data = sorted(zip(x, y))
    data = zip(*data)
    x, y = [list(t) for t in data]
    return x, y
    
def plot_training_data(logdir, datapath):
    X = np.load(os.path.join(datapath, 'X.npy'))
    X_indices = [0, 1, -1]
    X_names = ['x speed', 'y speed', 'yaw rate']
    Y = np.load(os.path.join(logdir, 'Y.npy'))
    Y_names = ['omega_o_' + str(i) for i in range(4)] + \
        ['mu_' + str(i) for i in range(4)]  + \
        ['z_R_' + str(i) for i in range(4)] + \
        ['z_I_' + str(i) for i in range(4)]
    fig, axes = plt.subplots(3, 16, figsize = (112, 21))
    for i in range(Y.shape[-1]):
        for j in range(3):
            y = Y[:, i].tolist()
            x = X[:, X_indices[j]].tolist()
            x, y = _sort(x, y)
            axes[j][i].scatter(Y[:, i], X[:, X_indices[j]])
            axes[j][i].set_ylabel(X_names[j])
            axes[j][i].set_xlabel(Y_names[i])
            axes[j][i].set_title(X_names[j] + ' vs ' + Y_names[i])
    fig.savefig(os.path.join(logdir, 'visualization.png'))


def _read_row(row, datapath):
    direction = row['direction']
    length = row['length']
    task = row['task']
    f = os.path.join(datapath, row['id'])
    y_items = ['omega_o', 'mu', 'z']
    x_items = ['achieved_goal']
    items = y_items + x_items
    data = {}
    for item in items:
        data[item] = np.load(f + '_' + item + '.npy')
    speed = np.sqrt(np.sum(np.square(
        np.mean(
            data['achieved_goal'][int(length * 0.25):],
            0
        )[:2]
    )))
    yaw = np.mean(data['achieved_goal'][int(length * 0.25):, -1])
    x = np.zeros(6, dtype = np.float32)
    y = np.concatenate([np.mean(data[item], 0) for item in y_items])
    if task == 'straight':
        if direction == 'forward':
            x[1] = speed
        elif direction == 'backward':
            x[1] = speed
        elif direction == 'left':
            x[0] = speed
        elif direction == 'right':
            x[0] = speed
        else:
            raise ValueError('Expected one of `forward`, `backward`, \
                    `left` or `right`, got {}'.format(direction))
    elif task == 'turn':
        x[0] = np.mean(data['achieved_goal'][int(length * 0.25):, 0], 0)
        x[1] = np.mean(data['achieved_goal'][int(length * 0.25):, 1], 0)
        x[-1] = np.mean(data['achieved_goal'][int(length * 0.25):, -1], 0)
    else:
        raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                got {}'.format(task))
    return x, y

def plot_training_data_v2(logdir, datapath):
    info = pd.read_csv(os.path.join(logdir, 'info.csv'), index_col = 0)
    tasks = ['straight', 'turn', 'rotate']
    directions = ['left', 'right', 'forward', 'backward']
    gaits = ['trot', 'ls_crawl', 'ds_crawl']
    X_indices = [0, 1, -1]
    X_names = ['x speed', 'y speed', 'yaw rate']
    Y_names = ['omega_o_' + str(i) for i in range(4)] + \
        ['mu_' + str(i) for i in range(4)]  + \
        ['z_R_' + str(i) for i in range(4)] + \
        ['z_I_' + str(i) for i in range(4)]
    if os.path.exists(os.path.join(logdir, 'visualizations')):
        shutil.rmtree(os.path.join(logdir, 'visualizations'))
    os.mkdir(os.path.join(logdir, 'visualizations'))
    for task in tasks:
        for gait in gaits:
            for direction in directions:
                df = info[
                    (info['task'] == task) & \
                    (info['direction'] == direction) & \
                    (info['gait'] == gait)
                ]
                X = []
                Y = []
                if len(df) == 0:
                    continue
                for i, row in df.iterrows():
                    x, y = _read_row(row, datapath)
                    X.append(x.copy())
                    Y.append(y.copy())
                X = np.stack(X, 0)
                Y = np.stack(Y, 0)
                fig, axes = plt.subplots(3, 16, figsize = (112, 21))
                for i in range(Y.shape[-1]):
                    for j in range(3):
                        y = Y[:, i].tolist()
                        x = X[:, X_indices[j]].tolist()
                        x, y = _sort(x, y)
                        axes[j][i].scatter(y, x)
                        axes[j][i].set_ylabel(X_names[j])
                        axes[j][i].set_xlabel(Y_names[i])
                        axes[j][i].set_title(X_names[j] + ' vs ' + Y_names[i])
                fig.savefig(
                    os.path.join(
                        logdir,
                        'visualizations',
                        '{}_{}_{}.png'.format(
                            task,
                            gait,
                            direction
                        )
                    )
                )

def _read_row_v2(row, datapath, tracklist):
    direction = row['direction']
    length = row['length']
    task = row['task']
    f = os.path.join(datapath, row['id'])
    y_items = ['omega_o', 'mu', 'z']
    x_items = ['achieved_goal']
    items = y_items + x_items + tracklist
    data = {}
    for item in items:
        data[item] = np.load(f + '_' + item + '.npy')
    speed = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    data['achieved_goal'][int(length * 0.25):, :2]
                ),
                -1
            )
        )
    )
    tracked = np.array(
        [
            np.mean(data[item][int(length * 0.25):], 0)[0] \
                for item in tracklist
        ],
        dtype = np.float32
    )
    yaw = np.mean(data['achieved_goal'][int(length * 0.25):, -1])
    x = np.zeros(6, dtype = np.float32)
    y = np.concatenate([np.mean(data[item], 0) for item in y_items])
    if task == 'straight':
        if direction == 'forward':
            x[1] = speed
        elif direction == 'backward':
            x[1] = speed
        elif direction == 'left':
            x[0] = speed
        elif direction == 'right':
            x[0] = speed
        else:
            raise ValueError('Expected one of `forward`, `backward`, \
                    `left` or `right`, got {}'.format(direction))
    elif task == 'turn':
        #x[0] = np.mean(data['achieved_goal'][int(length * 0.25):, 0], 0)
        x[1] = speed#np.mean(data['achieved_goal'][int(length * 0.25):, 1], 0)
        x[-1] = np.mean(data['achieved_goal'][int(length * 0.25):, -1], 0)
    elif task == 'rotate':
        x[-1] = np.mean(data['achieved_goal'][int(length * 0.25):, -1], 0)
    else:
        raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                got {}'.format(task))
    return x, y, tracked

def plot_training_data_v3(logdir, datapath,
        tracklist = ['d1', 'd2', 'd3', 'stability']
    ):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    tasks = ['straight', 'turn', 'rotate']
    directions = ['left', 'right', 'forward', 'backward']
    gaits = ['trot', 'ls_crawl', 'ds_crawl']
    X_indices = [0, 1, -1] 
    X_names = ['x speed', 'y speed', 'yaw rate']
    Y_names = ['omega_o_' + str(i) for i in range(4)] + \
        ['mu_' + str(i) for i in range(4)]  + \
        ['z_R_' + str(i) for i in range(4)] + \
        ['z_I_' + str(i) for i in range(4)]
    if os.path.exists(os.path.join(logdir, 'visualizations')):
        shutil.rmtree(os.path.join(logdir, 'visualizations'))
    os.mkdir(os.path.join(logdir, 'visualizations'))
    for task in tasks:
        os.mkdir(os.path.join(
            logdir,
            'visualizations',
            task
        ))
        for gait in gaits:
            os.mkdir(os.path.join(
                logdir,
                'visualizations',
                task,
                gait
            ))
            for direction in directions:
                os.mkdir(os.path.join(
                    logdir,
                    'visualizations',
                    task,
                    gait,
                    direction
                ))

    for task in tasks:
        for gait in gaits:
            for direction in directions:
                path = os.path.join(
                    logdir,
                    'visualizations',
                    task,
                    gait,
                    direction
                )
                df = info[
                    (info['task'] == task) & \
                    (info['direction'] == direction) & \
                    (info['gait'] == gait)
                ]
                X = []
                Y = []
                S = []
                X_ = []
                Y_ = []
                S_ = []
                if len(df) == 0:
                    continue
                for i, row in df.iterrows():
                    x, y, s = _read_row_v2(row, datapath, tracklist)
                    if i % 2 == 0:
                        X.append(x.copy())
                        Y.append(y.copy())
                        S.append(s.copy())
                    else:
                        X_.append(x.copy())
                        Y_.append(y.copy())
                        S_.append(s.copy())
                X = np.stack(X, 0)
                Y = np.stack(Y, 0)
                S = np.stack(S, 0)
                X_ = np.stack(X_, 0)
                Y_ = np.stack(Y_, 0)
                S_ = np.stack(S_, 0)
                fig, axes = plt.subplots(3, 16, figsize = (112, 21))
                for i in range(Y.shape[-1]):
                    for j in range(3):
                        y = Y[:, i].tolist()
                        x = X[:, X_indices[j]].tolist()
                        x, y = _sort(x, y)
                        axes[j][i].scatter(y, x)
                        axes[j][i].set_ylabel(X_names[j])
                        axes[j][i].set_xlabel(Y_names[i])
                        axes[j][i].set_title(X_names[j] + ' vs ' + Y_names[i])
                fig.savefig(
                    os.path.join(
                        path,
                        'data.png'
                    )
                )
                plt.close()

                fig, axes = plt.subplots(3, 16, figsize = (112, 21))
                for i in range(Y_.shape[-1]):
                    for j in range(3):
                        y = Y_[:, i].tolist()
                        x = X_[:, X_indices[j]].tolist()
                        x, y = _sort(x, y)
                        axes[j][i].scatter(y, x)
                        axes[j][i].set_ylabel(X_names[j])
                        axes[j][i].set_xlabel(Y_names[i])
                        axes[j][i].set_title(X_names[j] + ' vs ' + Y_names[i])
                fig.savefig(
                    os.path.join(
                        path,
                        'data_.png'
                    )
                )
                plt.close()

                fig, axes = plt.subplots(4, 16, figsize = (112, 28))
                for i in range(Y.shape[-1]):
                    for j in range(S.shape[-1]):
                        y = Y[:, i].tolist()
                        x = S[:, j].tolist()
                        x, y = _sort(x, y)
                        axes[j][i].scatter(y, x)
                        axes[j][i].set_ylabel(tracklist[j])
                        axes[j][i].set_xlabel(Y_names[i])
                        axes[j][i].set_title(tracklist[j] + ' vs ' + Y_names[i])
                fig.savefig(
                    os.path.join(
                        path,
                        'stability.png'
                    )
                )
                plt.close()

                fig, axes = plt.subplots(4, 16, figsize = (112, 28))
                for i in range(Y_.shape[-1]):
                    for j in range(S_.shape[-1]):
                        y = Y_[:, i].tolist()
                        x = S_[:, j].tolist()
                        x, y = _sort(x, y)
                        axes[j][i].scatter(y, x)
                        axes[j][i].set_ylabel(tracklist[j])
                        axes[j][i].set_xlabel(Y_names[i])
                        axes[j][i].set_title(tracklist[j] + ' vs ' + Y_names[i])
                fig.savefig(
                    os.path.join(
                        path,
                        'stability_.png'
                    )
                )
                plt.close()
                print('Find output files in: {}'.format(path))


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_train_eval(
    logdir, filename,
    train_loss, steps_train_loss, eval_loss, steps_eval_loss, weight
):
    train_loss = smooth(train_loss, weight)
    eval_loss = smooth(eval_loss, weight)
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    ax.plot(steps_train_loss, train_loss, 'b--', label = 'train loss')
    ax.plot(steps_eval_loss, eval_loss, 'r--', label = 'eval loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('Training Results')
    ax.legend()
    fig.savefig(os.path.join(logdir, filename))


def plot_ep_goal(logdir, datapath, filename = 'goal_comparison.png'):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'))
    index = np.random.randint(0, len(info))
    print(info.iloc[index])
    goal = np.load(os.path.join(
        datapath,
        'Quadruped_{}_achieved_goal.npy'.format(index)
    ))
    data = []
    T = np.arange(goal.shape[0]) * params['dt']
    for i in range(goal.shape[0]):
        if i + 1 < params['window_size'] // 2:
            data.append(
                np.mean(
                    goal[
                        :params['window_size'] // 2 + i + 1
                    ], 0
                )
            )
        elif i + 1 < goal.shape[0] - params['window_size'] // 2:
            data.append(
                np.mean(goal[
                    i + 1 - params['window_size'] // 2: i + 1 + params['window_size'] // 2
                ], 0)
            )
        else:
            data.append(
                np.mean(goal[
                    i + 1 - params['window_size']:
                ], 0)
            ) 
    data = np.stack(data, 0)
    fig, axes = plt.subplots(2, 3, figsize = (18, 12))
    axes[0][0].plot(T, goal[:, 0], '--b', label = 'achieved goal')
    axes[0][0].plot(T, data[:, 0], '--r', label = 'desired goal')
    axes[0][0].set_xlabel('time')
    axes[0][0].set_ylabel('x speed')
    axes[0][0].legend()
    axes[0][1].plot(T, goal[:, 1], '--b', label = 'achieved goal')
    axes[0][1].plot(T, data[:, 1], '--r', label = 'desired goal')
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('y speed')
    axes[0][1].legend()
    axes[0][2].plot(T, goal[:, 2], '--b', label = 'achieved goal')
    axes[0][2].plot(T, data[:, 2], '--r', label = 'desired goal')
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('z speed')
    axes[0][2].legend()
    axes[1][0].plot(T, goal[:, 3], '--b', label = 'achieved goal')
    axes[1][0].plot(T, data[:, 3], '--r', label = 'desired goal')
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('roll rate')
    axes[1][0].legend()
    axes[1][1].plot(T, goal[:, 4], '--b', label = 'achieved goal')
    axes[1][1].plot(T, data[:, 4], '--r', label = 'desired goal')
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('pitch rate')
    axes[1][1].legend()
    axes[1][2].plot(T, goal[:, 5], '--b', label = 'achieved goal')
    axes[1][2].plot(T, data[:, 5], '--r', label = 'desired goal')
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('yaw rate')
    axes[1][2].legend()
    fig.savefig(os.path.join(logdir, filename))
    plt.show()

def plot_ep_goal_v2(logdir, datapath, filename = 'goal_comparison.png'):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'))
    index = np.random.randint(0, len(info))
    print(info.iloc[index])
    goal = np.load(os.path.join(
        datapath,
        'Quadruped_{}_achieved_goal.npy'.format(index)
    ))  
    data = []
    T = np.arange(goal.shape[0]) * params['dt']
    params['window_size'] = int(goal.shape[0] * (11 / 24))
    for i in range(goal.shape[0]):
        if i + 1 < params['window_size'] // 2:
            data.append(
                np.mean(
                    goal[
                        :params['window_size'] // 2 + i + 1 
                    ], 0
                )   
            )   
        elif i + 1 < goal.shape[0] - params['window_size'] // 2:
            data.append(
                np.mean(goal[
                    i + 1 - params['window_size'] // 2: i + 1 + params['window_size'] // 2
                ], 0)
            )   
        else:
            data.append(
                np.mean(goal[
                    i + 1 - params['window_size']:
                ], 0)
            )
    data = np.stack(data, 0)
    fig, axes = plt.subplots(2, 3, figsize = (18, 12))
    axes[0][0].plot(T, goal[:, 0], '--b', label = 'achieved goal')
    axes[0][0].plot(T, data[:, 0], '--r', label = 'desired goal')
    axes[0][0].set_xlabel('time')
    axes[0][0].set_ylabel('x speed')
    axes[0][0].legend()
    axes[0][1].plot(T, goal[:, 1], '--b', label = 'achieved goal')
    axes[0][1].plot(T, data[:, 1], '--r', label = 'desired goal')
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('y speed')
    axes[0][1].legend()
    axes[0][2].plot(T, goal[:, 2], '--b', label = 'achieved goal')
    axes[0][2].plot(T, data[:, 2], '--r', label = 'desired goal')
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('z speed')
    axes[0][2].legend()
    axes[1][0].plot(T, goal[:, 3], '--b', label = 'achieved goal')
    axes[1][0].plot(T, data[:, 3], '--r', label = 'desired goal')
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('roll rate')
    axes[1][0].legend()
    axes[1][1].plot(T, goal[:, 4], '--b', label = 'achieved goal')
    axes[1][1].plot(T, data[:, 4], '--r', label = 'desired goal')
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('pitch rate')
    axes[1][1].legend()
    axes[1][2].plot(T, goal[:, 5], '--b', label = 'achieved goal')
    axes[1][2].plot(T, data[:, 5], '--r', label = 'desired goal')
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('yaw rate')
    axes[1][2].legend()
    fig.savefig(os.path.join(logdir, filename))
    plt.show()

def plot_z(steps = 1500, filename1 = 'assets/out/plots/temp.png',
        filename2 = 'assets/out/plots/temp2.png'):
    total = 6599
    index = np.random.randint(low = 0, high = 6599)
    logdir = 'assets/out/results_v9'
    f = 'Quadruped_{}_z.npy'.format(index)
    f2 = 'Quadruped_{}_omega_o.npy'.format(index)
    Z = np.load(os.path.join(logdir, f))
    omega = np.load(os.path.join(logdir, f2))
    omega = np.abs(omega[-1]) * 4
    T = np.repeat(np.expand_dims(np.arange(Z.shape[0]), -1), Z.shape[-1]//2, -1)* 0.001
    F = np.concatenate([
        np.cos(omega * T),
        np.sin(omega * T)
    ], -1)
    num_osc = Z.shape[-1] // 2
    fig, ax = plt.subplots(num_osc, 4, figsize = (40, num_osc * 10))
    colors = [np.random.random((3,)) for j in range(num_osc)]
    phase = np.arctan2(Z[:, :num_osc], Z[:, num_osc:])
    phase_ref = np.arctan2(F[:, :num_osc], F[:, num_osc:])
    diff = np.concatenate([
        np.expand_dims((phase[:, i] - phase_ref[:, i]) / (2 * np.pi), -1) for i in range(num_osc)
    ], -1)
    out = []
    heading_ctrl = [0, 1, 1, 1]
    for i in range(num_osc):
        arr = []
        for j in range(Z.shape[0]):
            if diff[j][i] < 0:
                arr.append(diff[j][i] + 1 * heading_ctrl[i])
            else:
                arr.append(diff[j][i])
        out.append(np.expand_dims(np.array(arr), -1))
    ref_phase_diff = np.concatenate(out, -1)
    ref_phase_diff = diff.copy()
    for i in range(num_osc):
        ax[i][0].plot(
            Z[-steps:, i],
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(str(i))
        )
        ax[i][0].plot(
            F[-steps:, i],
            color = 'r',
            linestyle = ':',
            label = 'ref'
        )
        ax[i][1].plot(
            Z[-steps:, i + num_osc],
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(str(i))
        )
        ax[i][1].plot(
            F[-steps:, i + num_osc],
            color = 'r',
            linestyle = ':',
            label = 'ref'
        )
        ax[i][2].plot(
            np.arctan2(Z[-steps:, i + num_osc], Z[-steps:, i]),
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(str(i))
        )
        ax[i][2].plot(
            np.arctan2(F[-steps:, i + num_osc], F[-steps:, i]),
            color = 'r',
            linestyle = ':',
            label = 'ref'
        )
        ax[i][3].plot(
            ref_phase_diff[-steps:, i],
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(str(i))
        )
        ax[i][0].legend(loc = 'upper left')
        ax[i][1].legend(loc = 'upper left')
        ax[i][2].legend(loc = 'upper left')
        ax[i][3].legend(loc = 'upper left')
    fig.savefig(filename1)
    
    fig2, ax2 = plt.subplots(1, 2, figsize = (20, 10))
    diff = np.concatenate([
        np.expand_dims((phase[:, i] - phase[:, 0]) / (2 * np.pi), -1) for i in range(num_osc)
    ], -1)
    out = []
    heading_ctrl = [1, 1, 1, 1]
    for i in range(num_osc):
        arr = []
        for j in range(Z.shape[0]):
            if diff[j][i] < 0:
                arr.append(1 + diff[j][i] * 1 * heading_ctrl[i])
            else:
                arr.append(diff[j][i])
        out.append(np.expand_dims(np.array(arr), -1))
    out = np.concatenate(out, -1)
    for i in range(num_osc):
        ax2[0].plot(
            out[:, i],
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(i)
        )
        ax2[1].plot(
            ref_phase_diff[:, i],
            color = colors[i],
            linestyle = '--',
            label = 'dim {}'.format(i)
        )
    ax2[0].legend(loc = 'upper left')
    ax2[1].legend(loc = 'upper left')
    fig2.savefig(filename2)
    plt.show()

def complex_multiply(z1, z2):
    x1, y1 = np.split(z1, 2, -1)
    x2, y2 = np.split(z2, 2, -1)
    return np.concatenate([
        x2 * x1 - y2 * y1,
        x2 * y1 + x1 * y2
    ], -1)

def test_coupling():
    logdir = 'assets/out/results_v9'
    index = np.random.randint(low = 0, high = 6599)
    info = pd.read_csv(os.path.join(logdir, 'info.csv'))
    print(info.iloc[index])
    gait = info.iloc[index]['gait']
    beta = 0.75
    if 'crawl' not in gait:
        beta = 0.5
    f = 'Quadruped_{}_z.npy'.format(index)
    f1 = 'Quadruped_{}_omega_o.npy'.format(index)
    f2 ='Quadruped_{}_heading_ctrl.npy'.format(index)
    Z = np.load(os.path.join(logdir, f))
    omega = np.load(os.path.join(logdir, f1))
    heading_ctrl = np.load(os.path.join(logdir, f2))
    dt = 0.001
    phase = np.arctan2(
        Z[0, :Z.shape[-1]//2], Z[0, Z.shape[-1]//2:]
    ) / np.pi

    print(phase)
    out = []
    for i in range(Z.shape[-1] // 2):
        if phase[i] < 0:
            out.append((phase[i] + 1.0))
        else:
            out.append(phase[i])
    phase = np.array(out) / 2
    omega = np.abs(omega[-1, :])
    phase -= np.cos(phase) * 3 * (1 - beta) / 8
    print(phase)
    phase = np.concatenate([
        np.cos(phase * 2 * np.pi),
        np.sin(phase * 2 * np.pi)
    ])
    out = []
    C = _get_polynomial_coef(params['degree'], params['thresholds'], dt * 50) 
    z = np.random.random((Z.shape[-1],))
    r = np.sqrt(
        np.square(Z[:, :Z.shape[-1] //2]) + \
        np.square(Z[:, Z.shape[-1] //2:])
    )
    z1 = np.concatenate([
        np.ones(Z.shape[-1]//2),
        np.zeros(Z.shape[-1]//2)
    ], -1)
    Z1 = []
    mu = r[-1, :].copy()
    for i in range(Z.shape[0] * 10):
        z1 = hopf_simple_step(omega, mu, z1, dt)
        z, w = hopf_mod_step(omega, mu, z, C, params['degree'], dt)
        x, y = np.split(phase, 2, -1)
        x1, y1 = np.split(z1, 2, -1)
        z += params['coupling_strength'] * np.concatenate([
            x1 * x - y1 * y,
            x1 * y + y1 * x
        ], -1)
        Z1.append(z1.copy())
        out.append(z.copy())
    out = np.stack(out, 0)
    Z1 = np.stack(Z1, 0)
    h = 4
    w = 2
    fig, ax = plt.subplots(h,w,figsize = (w * 10, h * 10))
    colors = [np.random.random((3, )) for i in range(h)]
    steps = int(2 * np.pi / (np.max(np.abs(omega)) * dt))
    for i in range(h):
        for j in range(w):
            ax[i][j].plot(
                out[-steps:, 2 * i + j],
                color = colors[i],
                linestyle = '-',
                label = 'dim {}'.format(i)
            )
            ax[i][j].plot(
                Z1[-steps:, 2 * i + j],
                color = 'g',
                linestyle = ':',
                label = 'ref dim {}'.format(i)
            )
            ax[i][j].legend(loc = 'upper left')
            ax[i][j].legend(loc = 'upper left')
    plt.show()
