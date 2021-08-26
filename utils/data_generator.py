import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from constants import params

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
    X = []
    Y = []
    for index, row in tqdm(info.iterrows()):
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
        for step in range(0, length, length // params['data_gen_granularity']):
            x = np.zeros(6, dtype = np.float32)
            y = np.concatenate([data[item][step, :] for item in y_items])
            Y.append(y.copy())
            pos = []
            for i in range(params['memory_size']):
                if step - i * params['memory_size'] > 0:
                    pos.append(data['joint_pos'][step - i * params['memory_size']])
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
            X.append(np.concatenate([
                x.copy(),
                data['achieved_goal'][step],
                pos
            ], -1))
    Y = np.stack(Y, 0)
    X = np.stack(X, 0)
    with open(os.path.join(logdir, 'X.npy'), 'wb') as f:
        np.save(f, X)
    with open(os.path.join(logdir, 'Y.npy'), 'wb') as f:
        np.save(f, Y)

def generate_multi_goal_gait_data(log_dir, env_class, env_kwargs, gait_list, task_list, direction_list, track_list, env_name):
    from constants import params
    DATA = {key : [] for key in track_list}
    TOTAL_STEPS = 0
    PREV_TOTAL_STEPS = 0
    num_files = 0
    cases = {'gait' : [], 'task' : [], 'direction' : [], 'id' : [], 'length' : []}
    print('Starting Data Generation.')
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
                        print('TOTAL STEPS: {}'.format(TOTAL_STEPS))
                        ep += 1
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
                    for ep in tqdm(range(params['n_epochs'] // 2)):
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
    df = pd.DataFrame(cases)
    df.to_csv(os.path.join(log_dir, 'info.csv'))
    print('Total Steps: {}'.format(TOTAL_STEPS))
    print('Data Generation Done.')


def get_reference_info(log_dir, track_list, env_name):
    files = os.listdir(log_dir)
    num_data = int(len(files) / len(track_list))
    data_path = [{key : os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, i, key)) for key in track_list} for i in range(num_data)]
    return num_data, data_path
