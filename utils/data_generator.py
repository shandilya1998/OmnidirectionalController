import os
import numpy as np
from constants import params
import argparse

track_list = ['joint_pos', 'action', 'velocity', 'position', 'true_joint_pos', 'sensordata', 'qpos', 'qvel', 'achieved_goal', 'observation', 'desired_goal']

def generate_multi_goal_gait_data(log_dir, env_class, env_kwargs, gait_list, task_list, direction_list, track_list, env_name):
    DATA = {key : [] for key in track_list}
    TOTAL_STEPS = 0
    PREV_TOTAL_STEPS = 0
    num_files = 0
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
                        not (direction not in ['right', 'left'] and task == 'rotate'):
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
            print("------------------------")
            print('TOTAL STEPS: {}'.format(TOTAL_STEPS))
            print('Saved {} {}'.format(gait, task))
            print('Case Steps {}'.format(TOTAL_STEPS - PREV_TOTAL_STEPS))
            print('------------------------')
            PREV_TOTAL_STEPS += TOTAL_STEPS


def read_multi_goal_generated_gait_data(log_dir, track_list, env_name):
    files = os.listdir(log_dir)
