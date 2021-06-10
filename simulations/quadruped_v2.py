import argparse
import gym
import numpy as np
from constants import params
from gym.envs.mujoco import mujoco_env
import random
from gym import utils, error
import os
import mujoco_py
from collections import OrderedDict
from tempfile import TemporaryFile
from utils import convert_observation_to_space, data_generator
from simulations.quadruped import Quadruped
import pandas as pd

class QuadrupedV2(Quadruped):
    def __init__(self,
                 model_path = 'ant.xml',
                 frame_skip = 5,
                 render = False,
                 gait = 'trot',
                 task = 'straight',
                 direction = 'forward',
                 policy_type = 'MultiInputPolicy',
                 track_lst = ['desired_goal', 'joint_pos', 'action', 'velocity', 'position', 'true_joint_pos', 'sensordata', 'qpos', 'qvel', 'achieved_goal', 'observation'],
                 stairs = False,
                 verbose = 0):
        super(QuadrupedV2, self).__init__(model_path, frame_skip, render, gait, task, direction, policy_type, track_lst, stairs, verbose)

    def _set_action_space(self):
        self.ref_path = params['ref_path']
        self.ref_num, self.data_path = data_generator.get_reference_info(self.ref_path, self._track_lst, params['env_name'])
        self.ref_info = pd.read_csv(os.path.join(params['ref_path'], 'info.csv'), index_col = 0)
        self.ref_command = self.ref_info.sample().values[0]
        self.gait = self.ref_command[0]
        self.task = self.ref_command[1]
        self.direction = self.ref_command[2]
        self.ref_data = {key : np.load(os.path.join(params['ref_path'], self.ref_command[3] + '_{}.npy'.format(key))) for key in self._track_lst}
        self.command = np.mean(self.ref_data['desired_goal'], axis = 0)
        self.desired_goal = self.command
        self.achieved_goal = self.sim.data.qvel[:6].copy()
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]] * 4
        self._set_beta()
        self._set_leg_params()
        self._joint_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self._set_init_gamma() # set init leg phase
        self.gamma = self.init_gamma.copy()
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._action_dim = low.shape[-1]
        return self.action_space

    def reset(self):
        self.ref_command = self.ref_info.sample().values[0]
        self.gait = self.ref_command[0]
        self.task = self.ref_command[1]
        self.direction = self.ref_command[2]
        self.ref_data = {key : np.load(os.path.join(params['ref_path'], self.ref_command[3] + '_{}.npy'.format(key))) for key in self._track_lst}
        self.command = np.mean(self.ref_data['desired_goal'], axis = 0)
        self.desired_goal = self.command
        self.achieved_goal = self.sim.data.qvel[:6].copy()
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]] * 4
        self._set_beta()
        self._set_leg_params()
        self._joint_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self._set_init_gamma() # set init leg phase
        self._step = 0
        self._last_base_position = [0, 0, params['INIT_HEIGHT']]
        self.gamma = self.init_gamma.copy()
        self.sim.reset()
        self.ob = self.reset_model()
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]] * 4
        """
        if self.policy_type == 'MultiInputPolicy':
        """
        """
                modify this according to observation space
        """
        """
            self.achieved_goal = self.sim.data.qvel[:6].copy()
            self.command = random.choice(self.commands)
            if self.verbose > 0:
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
            self.desired_goal = self.command
        """
        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        self._reset_track_lst()

        return self.ob

    def _track_attr(self):
        """
            modify this according to need
        """
        self._track_item['joint_pos'].append(self.joint_pos.copy())
        self._track_item['action'].append(self.action.copy())
        self._track_item['velocity'].append(self.sim.data.qvel[:6].copy())
        self._track_item['position'].append(self.sim.data.qpos[:3].copy())
        self._track_item['true_joint_pos'].append(self.sim.data.qpos[-self._num_joints:].copy())
        self._track_item['sensordata'].append(self.sim.data.sensordata.copy())
        self._track_item['qpos'].append(self.sim.data.qpos.copy())
        self._track_item['qvel'].append(self.sim.data.qvel.copy())
        ob =  self._get_obs()
        self._track_item['achieved_goal'].append(ob['achieved_goal'].copy())
        self._track_item['observation'].append(ob['observation'].copy())
        self._track_item['desired_goal'].append(ob['desired_goal'].copy())

    def do_simulation(self, action, n_frames, callback=None):
        self.sim.data.ctrl[:] = action
        self.action = action.copy()
        self.joint_pos = action.copy()
        for _ in range(n_frames):
            self.sim.step()

        self.achieved_goal = self.sim.data.qvel[:6].copy()

        self.w = [0.15, 0.3, 0.24, 0.13, 0.06, 0.06, 0.06]
        reward_velocity = np.exp(-np.linalg.norm(self.achieved_goal[:3] - self.desired_goal[:3], axis = -1)) * self.w[0]
        reward_ctrl = (0.85 * np.exp(-np.linalg.norm(action - self.ref_data['joint_pos'][self._step, :])) + \
            0.15 * np.exp(-np.linalg.norm(action - self.ref_data['true_joint_pos'][self._step, :]))) * self.w[1]
        reward_position = np.exp(-np.linalg.norm(self.sim.data.qpos[:3] - self.ref_data['qpos'][self._step, :3], axis = -1)) * self.w[2]
        reward_orientation = np.exp(-np.linalg.norm(self.sim.data.qpos[3:7] * self.ref_data['qpos'][self._step, 3:7])) * self.w[3]
        reward_ang_vel = np.exp(-np.linalg.norm(self.achieved_goal[3:6] - self.desired_goal[3:6], axis = -1)) * self.w[4]
        reward_contact = np.exp(-5e-4 * np.sum( 
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))
        )) * self.w[5]
        reward_energy = np.exp(-5e-4 * np.square(np.linalg.norm(self.sim.data.actuator_force))) * self.w[6]

        self._step += 1
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.02 and state[2] <= 0.3
        done = not notdone
        if self._step >= self.ref_data['sensordata'].shape[0] - 2:
            done = True

        self._track_attr()

        reward = reward_ctrl + reward_position + reward_velocity + reward_orientation + reward_ang_vel + reward_contact + reward_energy
        info = {
            'reward_velocity' : reward_velocity,
            'reward_position' : reward_position,
            'reward_energy' : reward_energy,
            'reward_ang_vel' : reward_ang_vel,
            'reward_orientation' : reward_orientation,
            'reward_contact' : reward_contact,
            'reward_ctrl' : reward_ctrl,
            'reward' : reward,
        }

        return reward, done, info

