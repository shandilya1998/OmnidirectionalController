import os
import numpy as np
from constants import params
import argparse

track_list = ['joint_pos', 'action', 'velocity', 'position', 'true_joint_pos', 'sensordata', 'qpos', 'qvel', 'achieved_goal', 'observation', 'desired_goal']

def generate_multi_goal_gait_data(log_dir, env_class, env_kwargs, gait_list, task_list, direction_list, track_list, env_name):
    DATA = {key : [] for key in track_list}
    TOTAL_STEPS = 0
    for gait in gait_list:
        if not os.path.exists(os.path.join(log_dir, gait)):
            os.mkdir(os.path.join(log_dir, gait))
        for task in task_list:
            if not os.path.exists(os.path.join(log_dir, gait, task)):
                os.mkdir(os.path.join(log_dir, gait, task))
            for direction in direction_list:
                try:
                    if not os.path.exists(os.path.join(log_dir, gait, task, direction)):
                        os.mkdir(os.path.join(log_dir, gait, task, direction))
                    ep = 0
                    data_case = {key : [] for key in track_list}
                    env = env_class(
                        gait = gait,
                        task = task,
                        direction = direction,
                        track_lst = track_list,
                        **env_kwargs
                    )
                    while ep < params['n_epochs']:
                        ep += 1
                        ac = env.action_space.sample()
                        ep_steps = 0
                        done = False
                        while not done and env._step < params['total_steps']:
                            ob, reward, done, info = env.step(ac)
                            ep_steps += 1
                        data = {}
                        for item in track_list:
                            data[item] = env._track_item[item]
                        """
                            modify according to need
                        """
                        mean_qvel = sum(data['achieved_goal'].copy()) / len(data['sensordata'])
                        mean_qvel[np.array([2, 3, 4], dtype = np.int32)] = 0.0 * np.array([2, 3, 4], dtype = np.int32)
                        data['desired_goal'] = [mean_qvel.copy()] * len(data['sensordata'])
                        """
                            ------------------------
                        """
                        if env._step < params['total_steps']:
                            while env._step < params['total_steps']:
                                for item in track_list:
                                    data[item].append(data[item][-1].copy())
                                env._step += 1
                        TOTAL_STEPS += env._step
                        ob = env.reset()
                        for item in track_list:
                            data_case[item].append(np.stack(data[item][:params['total_steps']], axis = 0).copy())
                    for item in track_list:
                        with open(os.path.join(log_dir, gait, task, direction, '{}_{}.npy'.format(env_name, item)), 'wb') as f:
                            np.save(f, np.stack(data_case[item], axis = 0))
                    env.close()
                except AssertionError:
                    pass
            print("------------------------")
            print('TOTAL STEPS: {}'.format(TOTAL_STEPS))
            print('Saved {} {}'.format(gait, task))
            print('------------------------')

