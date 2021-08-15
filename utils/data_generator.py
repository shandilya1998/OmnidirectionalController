import os
import numpy as np
import argparse
import pandas as pd

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
    x_items = ['desired_goal']
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
                data['desired_goal'][int(length * 0.25):],
                0
            )[:2]
        )))
        yaw = np.mean(data['desired_goal'][int(length * 0.25):, -1])
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
            x[0] = np.mean(data['desired_goal'][int(length * 0.25):, 0], 0)
            x[1] = np.mean(data['desired_goal'][int(length * 0.25):, 1], 0)
            x[-1] = np.mean(data['desired_goal'][int(length * 0.25):, -1], 0)
        elif task == 'rotate':
            if direction == 'left':
                x[-1] = -yaw
            elif direction == 'right':
                x[-1] = yaw
            else:
                raise ValueError('Expected one of `left` or `right`, got \
                        {}'.format(direction))
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
                                mu = (np.random.random() * 1.5 ) % 1
                                omega = 3.55
                                ac = np.array([ omega / (2 * np.pi), mu])
                            elif env._action_dim == 4:
                                omega = 3.55
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
                                mu = 0.5
                                omega = 1.6
                                ac = np.array([omega / (2 * np.pi), mu])
                            elif env._action_dim == 4:
                                omega = 1.6
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
                        while not done and ep_steps < params['MAX_STEPS'] // 2:
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

def get_reference_info(log_dir, track_list, env_name):
    files = os.listdir(log_dir)
    num_data = int(len(files) / len(track_list))
    data_path = [{key : os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, i, key)) for key in track_list} for i in range(num_data)]
    return num_data, data_path


