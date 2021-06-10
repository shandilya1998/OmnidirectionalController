import os
import numpy as np
import argparse
import pandas as pd

track_list = ['joint_pos', 'action', 'velocity', 'position', 'true_joint_pos', 'sensordata', 'qpos', 'qvel', 'achieved_goal', 'observation', 'desired_goal']

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

def get_reference_info(log_dir, track_list, env_name):
    files = os.listdir(log_dir)
    num_data = int(len(files) / len(track_list))
    data_path = [{key : os.path.join(log_dir, '{}_{}_{}.npy'.format(env_name, i, key)) for key in track_list} for i in range(num_data)]
    return num_data, data_path


