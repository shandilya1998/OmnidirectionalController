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
from utils.torch_utils import convert_observation_to_space
from utils import data_generator
from simulations.quadruped import Quadruped
import pandas as pd

class QuadrupedV2(Quadruped):
    def __init__(self,
                 model_path = 'ant.xml',
                 frame_skip = 5,
                 render = False,
                 policy_type = 'MultiInputPolicy',
                 track_lst = [ 
                     'desired_goal', 'joint_pos', 'action',
                     'velocity', 'position', 'true_joint_pos',
                     'sensordata', 'qpos', 'qvel',
                     'achieved_goal', 'observation', 'heading_ctrl',
                     'omega', 'z', 'mu',
                     'd1', 'd2', 'd3',
                     'stability', 'omega_o', 'reward'
                 ],
                 stairs = False,
                 verbose = 0):
        super(QuadrupedV2, self).__init__(
            model_path = model_path,
            frame_skip = frame_skip,
            render = render,
            policy_type = policy_type,
            track_lst = track_lst,
            verbose = verbose
        )
        if 'crawl' in self.gait:
            self.omega = np.array([1.6] * self._num_legs)
        elif self.gait == 'trot':
            self.omega = np.array([4.3])
        self.mu = np.ones((self._num_legs,))

    def _set_behaviour(self, gait, task, direction):
        self.gait = gait
        self.task = task
        self.direction = direction
        self._create_command_lst()

    def __sample_behaviour(self):
        self.gait = random.choice(['trot', 'ls_crawl', 'ds_crawl'])
        if 'crawl' in self.gait:
            self.task = random.choice(['straight', 'turn', 'rotate'])
        else:
            self.task = random.choice(['straight', 'turn'])
        if self.task == 'turn' or self.task == 'rotate':
            self.direction = random.choice(['left', 'right'])
        else:
            self.direction = random.choice([
                'forward', 'backward',
                'left', 'right'
            ])
        self._create_command_lst()

    def _set_action_space(self):
        self.init_b = np.concatenate([self.joint_pos, self.sim.data.sensordata.copy()], -1)
        self._set_beta()
        self._set_leg_params()
        self._set_init_gamma() # set init leg phase
        self.gamma = self.init_gamma.copy()
        self.commands =  self._create_command_lst()
        self.command = random.choice(self.commands)
        self.desired_goal = self.command
        self.achieved_goal = self.sim.data.qvel[:6].copy()
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = low.min() * np.ones(low.shape, dtype = np.float32)
        high = high.max() * np.ones(high.shape, dtype = np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._action_dim = low.shape[-1]
        return self.action_space

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
        self._track_item['desired_goal'].append(ob['desired_goal'].copy())
        self._track_item['observation'].append(ob['observation'].copy())
        self._track_item['heading_ctrl'].append(self.heading_ctrl.copy())
        self._track_item['omega_o'].append(self.omega.copy())
        self._track_item['omega'].append(self.w.copy())
        self._track_item['z'].append(self.z.copy())
        self._track_item['mu'].append(self.mu.copy())
        self._track_item['d1'].append(np.array([self.d1], dtype = np.float32))
        self._track_item['d2'].append(np.array([self.d2], dtype = np.float32))
        self._track_item['d3'].append(np.array([self.d3], dtype = np.float32))
        self._track_item['stability'].append(np.array([self.stability], dtype = np.float32))
        self._track_item['reward'].append(np.array([self._reward], dtype = np.float32))

    def _get_obs(self):
        """
            modify this according to observation space
        """
        ob = {}
        if self.policy_type == 'MultiInputPolicy':
            ob = {
                'observation' : np.concatenate([
                    self.joint_pos,
                    self.sim.data.sensordata.copy()
                ], -1),
                'desired_goal' : self.desired_goal,
                'achieved_goal' : self.achieved_goal
            }
        else:
            ob = np.concatenate([self.joint_pos, self.sim.data.sensordata.copy()], -1)
        return ob

    def reset(self):
        self._step = 0
        self._last_base_position = [0, 0, params['INIT_HEIGHT']]
        self.gamma = self.init_gamma.copy()
        #self.z = np.concatenate([np.cos((self.init_gamma + params['offset']) * np.pi * 2), np.sin((self.init_gamma + params['offset']) * np.pi * 2)], -1)
        self.z = self._get_z()
        self.sim.reset()
        self.ob = self.reset_model()
        if self.policy_type == 'MultiInputPolicy':
            """
                modify this according to observation space
            """
            self.achieved_goal = self.sim.data.qvel[:6].copy()
            self.__sample_behaviour()
            self.command = random.choice(self.commands)
            if self.verbose > 0:
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
            self.desired_goal = self.command.copy()

        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        self.d1, self.d2, self.d3, self.stability, upright = self.calculate_stability_reward(self.desired_goal)
        self._reset_track_lst()
        self._track_attr()
        return self.ob

    def do_simulation(self, action, n_frames, callback=None):
        self.action = action
        self._frequency = None
        self._amplitude = None
        
        reward_distance = 0.0
        reward_velocity = 0.0
        reward_energy = 0.0
        penalty = 0.0
        done = False
        
        if self.verbose > 0:
            print(self._n_steps)

        """
            The joint position of this is the same as the action:
                No precessing required here
        """

        self.joint_pos = self.action
        posbefore = self.get_body_com("torso").copy()
        penalty = 0.0
        if np.isnan(self.joint_pos).any():
            self.joint_pos = np.nan_to_num(self.joint_pos)
            penalty += -1.0
        self.sim.data.ctrl[:] = self.joint_pos
        for _ in range(n_frames):
            self.sim.step()
        posafter = self.get_body_com("torso").copy()
        velocity = (posafter - posbefore) / self.dt
        ang_vel = self.sim.data.qvel[3:6]
        self.d1, self.d2, self.d3, self.stability, upright = self.calculate_stability_reward(self.desired_goal)
        if self.policy_type == 'MultiInputPolicy':
            """
                modify this according to observation space
            """
            if len(self._track_item['achieved_goal']) \
                    > params['window_size']:
                self.achieved_goal = sum([np.concatenate([
                    velocity,
                    ang_vel
                ], -1)] + self._track_item[
                    'achieved_goal'][-params['window_size'] + 1:
                    ]
                ) / params['window_size']
            else:
                self.achieved_goal = sum([np.concatenate([
                    velocity,
                    ang_vel
                ], -1)] + [self._track_item[
                    'achieved_goal'][0]] * (params['window_size'] - 1)
                ) / params['window_size']
        if self._is_render:
            self.render()
        if self.policy_type == 'MultiInputPolicy':
            reward_velocity += np.sum(np.square(
                self.achieved_goal - self.desired_goal
            ))
        else:
            reward_velocity += np.square(
                self.achieved_goal[0] - self.desired_goal[0]
            )
        reward_energy += 0.5 * np.linalg.norm(
            self.sim.data.actuator_force * self.sim.data.qvel[-self._num_joints:]
        ) + 0.5 * 1e-3 * np.sum(np.square(
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
        )
        if not upright:
            done = True
        self._track_attr()
        self._step += 1
        reward_distance = np.linalg.norm(self.sim.data.qpos[:2])
        reward_velocity = np.exp(-reward_velocity * 1e3)
        reward_energy = np.exp(-reward_energy)
        reward = reward_velocity + reward_energy + penalty + reward_distance
        info = {
            'reward_distance' : reward_distance,
            'reward_velocity' : reward_velocity,
            'reward_energy' : reward_energy,
            'reward' : reward,
            'penalty' : penalty
        }

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.02 and state[2] <= 0.5
        done = not notdone
        self._reward = reward

        if self._step >= params['max_episode_size']:
            done = True

        return reward, done, info

def test():
    index = np.random.randint(low = 0, high = 6599)
    logdir = 'assets/out/results_v9'
    info = pd.read_csv(os.path.join(logdir, 'info.csv'), index_col = 0)
    print(info.iloc[index])
    f1 = 'Quadruped_{}_joint_pos.npy'.format(index)
    joint_pos = np.load(os.path.join(logdir, f1))
    
    env = QuadrupedV2()

    for i in range(joint_pos.shape[0]):
        ob = env.step(joint_pos[i, :])
        env.render()

