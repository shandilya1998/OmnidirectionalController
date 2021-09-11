import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from constants import params
import shutil
import h5py
from simulations import Quadruped

track_list = [
    'joint_pos', 'action', 'velocity', \
    'position', 'true_joint_pos', 'sensordata', \
    'qpos', 'qvel', 'achieved_goal', \
    'observation', 'desired_goal', 'heading_ctrl', \
    'omega', 'z', 'mu', 'd1', 'd2', 'd3', \
    'stability', 'omega_o'
]

def create_training_data(logdir, datapath):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    y_items = ['omega_o', 'mu', 'z']
    x_items = ['achieved_goal']
    items = y_items + x_items
    X = []
    Y = []
    for index, row in info.iterrows():
        direction = row['direction']
        length = row['length']
        task = row['task']
        f = os.path.join(datapath, row['id'])
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
        Y.append(y.copy())
        if task == 'straight':
            if direction == 'forward':
                x[1] = speed
            elif direction == 'backward':
                x[1] = -speed
            elif direction == 'left':
                x[0] = -speed
            elif direction == 'right':
                x[0] = speed
            else:
                raise ValueError('Expected one of `forward`, `backward`, \
                        `left` or `right`, got {}'.format(direction))
        elif task == 'turn':
            x[0] = np.mean(data['achieved_goal'][int(length * 0.25):, 0], 0)
            x[1] = np.mean(data['achieved_goal'][int(length * 0.25):, 1], 0)
            x[-1] = np.mean(data['achieved_goal'][int(length * 0.25):, -1], 0)
        elif task == 'rotate':
            x[-1] = yaw
        else:
            raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                    got {}'.format(task))
        X.append(x.copy())
    Y = np.stack(Y, 0)
    X = np.stack(X, 0)
    with open(os.path.join(logdir, 'X.npy'), 'wb') as f:
        np.save(f, X)
    with open(os.path.join(logdir, 'Y.npy'), 'wb') as f:
        np.save(f, Y)

def create_training_data_v2(logdir, datapath):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    y_items = ['omega_o', 'mu', 'z']
    x_items = ['achieved_goal', 'joint_pos']
    items = y_items + x_items
    num_files = 0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if os.path.exists(os.path.join(logdir, 'temp')):
        shutil.rmtree(os.path.join(logdir, 'temp'))
    os.mkdir(os.path.join(logdir, 'temp'))
    count = 0
    for index, row in tqdm(info.iterrows()):
        X = []
        Y = []
        direction = row['direction']
        length = row['length']
        task = row['task']
        f = os.path.join(datapath, row['id'])
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
        steps = length // params['data_gen_granularity']
        if steps < 1:
            steps = 1
        for step in range(0, length, steps):
            x = np.zeros(6, dtype = np.float32)
            y = np.concatenate([data[item][step, :] for item in y_items])
            Y.append(y.copy())
            pos = []
            for i in range(params['memory_size']):
                if step - i * params['memory_size'] > 0:
                    pos.append(data['joint_pos'][step - i * params['memory_size'] - 1])
                else:
                    pos.append(data['joint_pos'][0])
            pos = np.concatenate(pos, -1)
            if task == 'straight':
                if direction == 'forward':
                    x[1] = speed
                elif direction == 'backward':
                    x[1] = -speed
                elif direction == 'left':
                    x[0] = -speed
                elif direction == 'right':
                    x[0] = speed
                else:
                    raise ValueError('Expected one of `forward`, `backward`, \
                            `left` or `right`, got {}'.format(direction))
            elif task == 'turn':
                x[-1] = yaw
                if step < params['window_size']:
                    x[0] = np.mean(data['achieved_goal'][:params['window_size'], 0], 0)
                    x[1] = np.mean(data['achieved_goal'][:params['window_size'], 1], 0)
                else:
                    x[0] = np.mean(data['achieved_goal'][step - params['window_size']: step, 0], 0)
                    x[1] = np.mean(data['achieved_goal'][step - params['window_size']: step, 1], 0)
            elif task == 'rotate':
                x[-1] = yaw
            else:
                raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                        got {}'.format(task))
            count += 1
            X.append(np.concatenate([
                x.copy(),
                data['achieved_goal'][step],
                pos
            ], -1))

        X = np.stack(X, 0)
        Y = np.stack(Y, 0)
        with open(os.path.join(logdir, 'temp', 'X_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, X.copy())
        with open(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, Y.copy())
        num_files += 1

    dataX = np.zeros((count, params['input_size_low_level_control']), dtype = np.float32)
    dataY = np.zeros((count, params['cpg_param_size']), dtype = np.float32)
    count = 0
    for i in tqdm(range(num_files)):
        x = np.load(os.path.join(logdir, 'temp', 'X_{}.npy'.format(i)))
        y = np.load(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(i)))
        for j in range(x.shape[0]):
            dataX[count] = x[j, :]
            dataY[count] = y[j, :]
            count += 1
    f = h5py.File(os.path.join(logdir, 'data.hdf5'), 'w')
    d1 = f.create_dataset('X', dataX.shape, dtype = 'f', data = dataX)
    d2 = f.create_dataset('Y', dataY.shape, dtype = 'f', data = dataY)
    d1.attrs['size'] = params['input_size_low_level_control']
    d2.attrs['size'] = params['cpg_param_size']
    shutil.rmtree(os.path.join(logdir, 'temp'))

def create_training_data_v3(logdir, datapath):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    y_items = ['omega_o', 'mu', 'z']
    x_items = ['achieved_goal', 'joint_pos', 'z']
    items = y_items + x_items
    num_files = 0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if os.path.exists(os.path.join(logdir, 'temp')):
        shutil.rmtree(os.path.join(logdir, 'temp'))
    os.mkdir(os.path.join(logdir, 'temp'))
    count = 0
    for index, row in tqdm(info.iterrows()):
        X = []
        Y = []
        direction = row['direction']
        length = row['length']
        task = row['task']
        f = os.path.join(datapath, row['id'])
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
        steps = length // params['data_gen_granularity']
        if steps < 1:
            steps = 1
        for step in range(0, length, steps):
            x = np.zeros(6, dtype = np.float32)
            y = np.concatenate([data[item][step, :] for item in y_items])
            Y.append(y.copy())
            pos = []
            z = []
            for i in range(params['memory_size']):
                if step - i * params['memory_size'] > 1:
                    pos.append(data['joint_pos'][step - i * params['memory_size'] - 1])
                    z.append(data['z'][step - i * params['memory_size'] - 1])
                else:
                    pos.append(data['joint_pos'][0])
                    z.append(data['z'][0])
                
            pos = np.concatenate(pos, -1)
            z = np.concatenate(z, -1)
            if task == 'straight':
                if direction == 'forward':
                    x[1] = speed
                elif direction == 'backward':
                    x[1] = -speed
                elif direction == 'left':
                    x[0] = -speed
                elif direction == 'right':
                    x[0] = speed
                else:
                    raise ValueError('Expected one of `forward`, `backward`, \
                            `left` or `right`, got {}'.format(direction))
            elif task == 'turn':
                x[-1] = yaw
                if step < params['window_size']:
                    x[0] = np.mean(data['achieved_goal'][:params['window_size'], 0], 0)
                    x[1] = np.mean(data['achieved_goal'][:params['window_size'], 1], 0)
                else:
                    x[0] = np.mean(data['achieved_goal'][step - params['window_size']: step, 0], 0)
                    x[1] = np.mean(data['achieved_goal'][step - params['window_size']: step, 1], 0)
            elif task == 'rotate':
                x[-1] = yaw
            else:
                raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                        got {}'.format(task))
            count += 1
            X.append(np.concatenate([
                x.copy(),
                data['achieved_goal'][step],
                pos,
                z
            ], -1))

        X = np.stack(X, 0)
        Y = np.stack(Y, 0)
        with open(os.path.join(logdir, 'temp', 'X_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, X.copy())
        with open(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, Y.copy())
        num_files += 1

    dataX = np.zeros((count, params['input_size_low_level_control']), dtype = np.float32)
    dataY = np.zeros((count, params['cpg_param_size']), dtype = np.float32)
    count = 0
    for i in tqdm(range(num_files)):
        x = np.load(os.path.join(logdir, 'temp', 'X_{}.npy'.format(i)))
        y = np.load(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(i)))
        for j in range(x.shape[0]):
            dataX[count] = x[j, :]
            dataY[count] = y[j, :]
            count += 1
    f = h5py.File(os.path.join(logdir, 'data.hdf5'), 'w')
    d1 = f.create_dataset('X', dataX.shape, dtype = 'f', data = dataX)
    d2 = f.create_dataset('Y', dataY.shape, dtype = 'f', data = dataY)
    d1.attrs['size'] = params['input_size_low_level_control']
    d2.attrs['size'] = params['cpg_param_size']
    shutil.rmtree(os.path.join(logdir, 'temp'))



def _get_weights(init_gamma):
    """
        0 - 0 to 1
        1 - 0 to 2
        2 - 0 to 3
        3 - 1 to 2
        4 - 1 to 3
        5 - 2 to 3
    """
    dim1 = [0, 0, 0, 1, 1, 2]
    dim2 = [1, 2, 3, 2, 3, 3]
    out = []
    for i in range(6):
        out.append(2 * np.pi * (init_gamma[dim1[i]] - init_gamma[dim2[i]]))
    return np.array(out, dtype = np.float32)

def create_training_data_v4(logdir, datapath):
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    y_items = ['omega_o', 'mu']
    x_items = ['achieved_goal', 'joint_pos', 'z']
    items = y_items + x_items
    num_files = 0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if os.path.exists(os.path.join(logdir, 'temp')):
        shutil.rmtree(os.path.join(logdir, 'temp'))
    os.mkdir(os.path.join(logdir, 'temp'))
    count = 0

    init_gamma = {}
    heading_ctrl = {}
    count = 0
    for gait in params['gait_list']:
        for task in params['task_list']:
            for direction in params['direction_list']:
                if not (gait not in ['ds_crawl', 'ls_crawl'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'turn'):
                    count += 1
                    env = Quadruped(
                        gait = gait,
                        task = task,
                        direction = direction,
                        track_lst = params['track_list']
                    )
                    ig, hc = env._get_init_gamma(
                        gait = gait,
                        task = task,
                        direction = direction
                    )
                    init_gamma[(gait, task, direction)] = ig.copy()
                    heading_ctrl[(gait, task, direction)] = hc.copy()

    for index, row in tqdm(info.iterrows()):
        X = []
        Y = []
        direction = row['direction']
        length = row['length']
        task = row['task']
        f = os.path.join(datapath, row['id'])
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
        steps = length // params['data_gen_granularity']
        if steps < 1:
            steps = 1
        for step in range(0, length, steps):
            x = np.zeros(6, dtype = np.float32)
            weights = _get_weights(
                init_gamma[(gait, task, direction)],
            )
            y = np.concatenate(
                [data[item][step, :] for item in y_items] + [weights], -1
            )
            Y.append(y.copy())
            pos = []
            z = []
            for i in range(params['memory_size']):
                if step - i * params['memory_size'] > 1:
                    pos.append(data['joint_pos'][step - i * params['memory_size'] - 1])
                    z.append(data['z'][step - i * params['memory_size'] - 1])
                else:
                    pos.append(data['joint_pos'][0])
                    z.append(data['z'][0])

            pos = np.concatenate(pos, -1)
            z = np.concatenate(z, -1)
            if task == 'straight':
                if direction == 'forward':
                    x[1] = speed
                elif direction == 'backward':
                    x[1] = -speed
                elif direction == 'left':
                    x[0] = -speed
                elif direction == 'right':
                    x[0] = speed
                else:
                    raise ValueError('Expected one of `forward`, `backward`, \
                            `left` or `right`, got {}'.format(direction))
            elif task == 'turn':
                x[-1] = yaw
                if step < params['window_size']:
                    x[0] = np.mean(data['achieved_goal'][:params['window_size'], 0], 0)
                    x[1] = np.mean(data['achieved_goal'][:params['window_size'], 1], 0)
                else:
                    x[0] = np.mean(data['achieved_goal'][step - params['window_size']: step, 0], 0)
                    x[1] = np.mean(data['achieved_goal'][step - params['window_size']: step, 1], 0)
            elif task == 'rotate':
                x[-1] = yaw
            else:
                raise ValueError('Expected one of `straight`, `turn` or `rotate`, \
                        got {}'.format(task))
            count += 1
            if params['observation_version'] == 1:
                X.append(np.concatenate([
                    x.copy(),
                    data['achieved_goal'][step],
                    pos,
                    z
                ], -1))
            elif params['observation_version'] == 0:
                X.append(np.concatenate([
                    x.copy(),
                    data['achieved_goal'][step],
                    pos,
                ], -1))

        X = np.stack(X, 0)
        Y = np.stack(Y, 0)
        with open(os.path.join(logdir, 'temp', 'X_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, X.copy())
        with open(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(num_files)), 'wb') as f:
            np.save(f, Y.copy())
        num_files += 1

    dataX = np.zeros((count, params['input_size_low_level_control']), dtype = np.float32)
    dataY = np.zeros((count, params['cpg_param_size']), dtype = np.float32)
    count = 0
    for i in tqdm(range(num_files)):
        x = np.load(os.path.join(logdir, 'temp', 'X_{}.npy'.format(i)))
        y = np.load(os.path.join(logdir, 'temp', 'Y_{}.npy'.format(i)))
        for j in range(x.shape[0]):
            dataX[count] = x[j, :]
            dataY[count] = y[j, :]
            count += 1
    f = h5py.File(os.path.join(logdir, 'data.hdf5'), 'w')
    d1 = f.create_dataset('X', dataX.shape, dtype = 'f', data = dataX)
    d2 = f.create_dataset('Y', dataY.shape, dtype = 'f', data = dataY)
    d1.attrs['size'] = params['input_size_low_level_control']
    d2.attrs['size'] = params['cpg_param_size']
    shutil.rmtree(os.path.join(logdir, 'temp'))


def generate_multi_goal_gait_data(log_dir, env_class, env_kwargs, gait_list, task_list, direction_list, track_list, env_name):
    from constants import params
    DATA = {key : [] for key in track_list}
    TOTAL_STEPS = 0
    PREV_TOTAL_STEPS = 0
    num_files = 0
    cases = {'gait' : [], 'task' : [], 'direction' : [], 'id' : [], 'length' : []}
    print('Starting Data Generation.')
    count = 0
    for gait in gait_list:
        for task in task_list:
            for direction in direction_list:
                ep = 0
                while ep < params['n_epochs'] and \
                    not (gait not in ['ds_crawl', 'ls_crawl'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'turn'):
                    count += 1
                    ep += 1
    pbar = tqdm(total=count)
    for gait in gait_list:
        for task in task_list:
            for direction in direction_list:
                try:
                    ep = 0
                    data_case = {key : [] for key in track_list}
                    env = env_class(
                        gait = gait,
                        task = task,
                        direction = direction,
                        track_lst = track_list,
                        **env_kwargs
                    )
                    while ep < params['n_epochs'] and \
                        not (gait not in ['ds_crawl', 'ls_crawl'] and task == 'rotate') and \
                        not (direction not in ['right', 'left'] and task == 'rotate') and \
                        not (direction not in ['right', 'left'] and task == 'turn'):
                        ep += 1
                        pbar.update(1)
                        ac = env.action_space.sample()
                        if env.gait == 'trot':
                            if env._action_dim == 2:
                                mu = np.random.random()
                                omega = np.random.uniform(
                                    low = np.pi,
                                    high = 11 * np.pi / 6 - np.pi / 10
                                )
                                ac = np.array([ omega / (2 * np.pi), mu])
                            elif env._action_dim == 4:
                                omega = np.random.uniform(
                                    low = np.pi,
                                    high = 11 * np.pi / 6 - np.pi / 10
                                ) 
                                mu1 = np.random.random()
                                mu2 = np.random.uniform(low = mu1, high = 1.0)
                                if env.direction == 'left':
                                    mu = np.array([mu2, mu1], dtype = np.float32)
                                    ac = np.array([(omega) / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
                                elif env.direction == 'right':
                                    mu = np.array([mu1, mu2], dtype = np.float32)
                                    ac = np.array([omega / (2 * np.pi), mu[0], (omega) / (2 * np.pi), mu[1]], dtype = np.float32)
                        if 'crawl' in env.gait:
                            if env._action_dim == 2:
                                mu = np.random.random()
                                omega = np.random.uniform(
                                    low = np.pi / 6,
                                    high = 5 * np.pi / 6 
                                ) 
                                ac = np.array([omega / (2 * np.pi), mu])
                            elif env._action_dim == 4:
                                omega = np.random.uniform(
                                    low = np.pi / 6,
                                    high = 5 * np.pi / 6 
                                ) 
                                mu1 = np.random.random()
                                mu2 = np.random.uniform(low = mu1, high = 1.0)
                                if env.direction == 'left':
                                    mu = np.array([mu2, mu1], dtype = np.float32)
                                    ac = np.array([omega / (2 * np.pi), mu[0], (omega)/ (2 * np.pi), mu[1]], dtype = np.float32)
                                elif env.direction == 'right':
                                    mu = np.array([mu1, mu2], dtype = np.float32)
                                    ac = np.array([omega / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
                        ep_steps = 0
                        done = False
                        while not done and ep_steps < params['MAX_STEPS']:
                            ob, reward, done, info = env.step(ac)
                            ep_steps += 1

                        data = {}
                        for item in track_list:
                            data[item] = env._track_item[item]
                        """
                            modify according to need
                        """
                        data['desired_goal'] = []
                        for i in range(len(data['sensordata'])):
                            if i + 1 <= 100 :
                                mean_qvel = sum(data['achieved_goal'][:i+1]) / (i + 1)
                            else:
                                mean_qvel = sum(data['achieved_goal'][i+1-100:i+1]) / 100
                            mean_qvel[np.array([2, 3, 4], dtype = np.int32)] = 0.0 * np.array([2, 3, 4], dtype = np.int32)
                            data['desired_goal'].append(mean_qvel.copy())
                        cases['gait'].append(gait)
                        cases['task'].append(task)
                        cases['direction'].append(direction)
                        cases['id'].append('{}_{}'.format(env_name, num_files))
                        cases['length'].append(len(data['sensordata']))
                        TOTAL_STEPS += len(data['sensordata'])
                        """
                            ------------------------
                        """
                        for item in track_list:
                            with open(os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, num_files, item)), 'wb') as f:
                                np.save(f, np.stack(data[item], axis = 0))
                        num_files += 1
                        env.reset()
                    env.close()
                except AssertionError:
                    pass
    pbar.close()
    print('TOTAL STEPS: {}'.format(TOTAL_STEPS))
    df = pd.DataFrame(cases)
    df.to_csv(os.path.join(log_dir, 'info.csv'))
    print('Data Generation Done.')


def generate_multi_goal_gait_data_v2(log_dir, env_class, env_kwargs, gait_list, task_list, direction_list, track_list, env_name):
    from constants import params
    DATA = {key : [] for key in track_list}
    TOTAL_STEPS = 0 
    PREV_TOTAL_STEPS = 0 
    num_files = 0 
    cases = {'gait' : [], 'task' : [], 'direction' : [], 'id' : [], 'length' : []} 
    print('Starting Data Generation.')
    props = {
        'ls_crawl' : {
            'omega' : [np.pi / 6, 5 * np.pi / 6],
            'mu' : [0, 1]
        },
        'ds_crawl' : {  
            'omega' : [np.pi / 6, 5 * np.pi / 6],
            'mu' : [0, 1]
        },
        'trot' : {
            'omega' : [np.pi, 11 * np.pi / 6 - np.pi / 10],
            'mu' : [0, 1]
        },

    }
    count = 0
    for gait in gait_list:
        for task in task_list:
            for direction in direction_list:
                ep = 0
                while ep < params['n_epochs'] and \
                    not (gait not in ['ds_crawl', 'ls_crawl'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'turn'):
                    count += 2
                    ep += 1
    pbar = tqdm(total=count)
    for gait in gait_list:
        for task in task_list:
            for direction in direction_list:
                if not (gait not in ['ds_crawl', 'ls_crawl'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'rotate') and \
                    not (direction not in ['right', 'left'] and task == 'turn'):
                    ep = 0
                    data_case = {key : [] for key in track_list}
                    env = env_class(
                        gait = gait,
                        task = task,
                        direction = direction,
                        track_lst = track_list,
                        **env_kwargs
                    )
                    for ep in range(params['n_epochs']):
                        pbar.update(1)
                        ac = env.action_space.sample()
                        # Constant omega 
                        if env.gait == 'trot':
                            omega = 3.55
                        elif 'crawl' in env.gait:
                            omega = 1.6
                        if env._action_dim == 2:
                            mu = np.random.random()
                            ac = np.array([omega / (2 * np.pi), mu])
                        elif env._action_dim == 4:
                            mu1 = np.random.random()
                            mu2 = np.random.uniform(low = mu1, high = 1.0)
                            if env.direction == 'left':
                                mu = np.array([mu2, mu1], dtype = np.float32)
                                ac = np.array([(omega) / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
                            elif env.direction == 'right':
                                mu = np.array([mu1, mu2], dtype = np.float32)
                                ac = np.array([omega / (2 * np.pi), mu[0], (omega) / (2 * np.pi), mu[1]], dtype = np.float32)     

                        ep_steps = 0 
                        done = False
                        while not done and ep_steps < params['MAX_STEPS']:
                            ob, reward, done, info = env.step(ac)
                            ep_steps += 1

                        data = {}
                        for item in track_list:
                            data[item] = env._track_item[item]
                        """ 
                            modify according to need
                        """
                        data['desired_goal'] = []
                        for i in range(len(data['sensordata'])):
                            if i + 1 <= 100 :
                                mean_qvel = sum(data['achieved_goal'][:i+1]) / (i + 1)
                            else:
                                mean_qvel = sum(data['achieved_goal'][i+1-100:i+1]) / 100 
                            mean_qvel[np.array([2, 3, 4], dtype = np.int32)] = 0.0 * np.array([2, 3, 4], dtype = np.int32)
                            data['desired_goal'].append(mean_qvel.copy())
                        cases['gait'].append(gait)
                        cases['task'].append(task)
                        cases['direction'].append(direction)
                        cases['id'].append('{}_{}'.format(env_name, num_files))
                        cases['length'].append(len(data['sensordata']))
                        TOTAL_STEPS += len(data['sensordata'])
                        """ 
                            ------------------------
                        """
                        for item in track_list:
                            with open(os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, num_files, item)), 'wb') as f:
                                np.save(f, np.stack(data[item], axis = 0)) 
                        num_files += 1
                        env.reset()

                        # Constant mu
                        omega = np.random.uniform(
                            low = props[env.gait]['omega'][0],
                            high = props[env.gait]['omega'][1]
                        )
                        if env._action_dim == 2:
                            mu = 0.5
                            ac = np.array([omega / (2 * np.pi), mu])
                        elif env._action_dim == 4:
                            mu1 = 0.33
                            mu2 = 0.67
                            if env.direction == 'left':
                                mu = np.array([mu2, mu1], dtype = np.float32)
                                ac = np.array([(omega) / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
                            elif env.direction == 'right':
                                mu = np.array([mu1, mu2], dtype = np.float32)
                                ac = np.array([omega / (2 * np.pi), mu[0], (omega) / (2 * np.pi), mu[1]], dtype = np.float32)

                        ep_steps = 0 
                        done = False
                        pbar.update(1)
                        while not done and ep_steps < params['MAX_STEPS']:
                            ob, reward, done, info = env.step(ac)
                            ep_steps += 1

                        data = {}
                        for item in track_list:
                            data[item] = env._track_item[item]
                        """ 
                            modify according to need
                        """
                        data['desired_goal'] = []
                        for i in range(len(data['sensordata'])):
                            if i + 1 <= 100 :
                                mean_qvel = sum(data['achieved_goal'][:i+1]) / (i + 1)
                            else:
                                mean_qvel = sum(data['achieved_goal'][i+1-100:i+1]) / 100 
                            mean_qvel[np.array([2, 3, 4], dtype = np.int32)] = 0.0 * np.array([2, 3, 4], dtype = np.int32)
                            data['desired_goal'].append(mean_qvel.copy())
                        cases['gait'].append(gait)
                        cases['task'].append(task)
                        cases['direction'].append(direction)
                        cases['id'].append('{}_{}'.format(env_name, num_files))
                        cases['length'].append(len(data['sensordata']))
                        TOTAL_STEPS += len(data['sensordata'])
                        """ 
                            ------------------------
                        """
                        for item in track_list:
                            with open(os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, num_files, item)), 'wb') as f:
                                np.save(f, np.stack(data[item], axis = 0)) 
                        num_files += 1
                        env.reset()
                    env.close()
    pbar.close()
    df = pd.DataFrame(cases)
    df.to_csv(os.path.join(log_dir, 'info.csv'))
    print('Total Steps: {}'.format(TOTAL_STEPS))
    print('Data Generation Done.')

def get_reference_info(log_dir, track_list, env_name):
    files = os.listdir(log_dir)
    num_data = int(len(files) / len(track_list))
    data_path = [{key : os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, i, key)) for key in track_list} for i in range(num_data)]
    return num_data, data_path
