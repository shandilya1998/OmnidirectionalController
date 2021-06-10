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
        super(QuadrupedV2, self).__init__(model_path, frame_skip, gait, task, direction, policy_type, track_lst, stairs, verbose)

    def _set_action_space(self):
        self.ref_path = params['ref_path']
        self.num_ref, self.data_path = data_generator.get_reference_info(self.ref_path, self.track_lst, params['env_name'])
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]] * 4
        self._set_beta()
        self._set_leg_params()
        self._joint_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self._set_init_gamma() # set init leg phase
        self.gamma = self.init_gamma.copy()
        if self.policy_type == 'MultiInputPolicy':
            self.commands =  self._create_command_lst()
            self.command = random.choice(self.commands)
            self.desired_goal = self.command
            self.achieved_goal = self.sim.data.qvel[:6].copy()

        if self.task != 'turn':
            low = np.ones((2,), dtype = np.float32) * 0.005
            high = np.ones((2,), dtype = np.float32)
            self._action_dim = 2
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            return self.action_space
        else:
            self._action_dim = 4
            low = np.ones((4,), dtype = np.float32) * 0.005
            high = np.ones((4,), dtype = np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            return self.action_space

    def reset(self):
        self._step = 0
        self._last_base_position = [0, 0, params['INIT_HEIGHT']]
        self.gamma = self.init_gamma.copy()
        self.sim.reset()
        self.ob = self.reset_model()
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]] * 4
        if self.policy_type == 'MultiInputPolicy':
            """
                modify this according to observation space
            """
            self.achieved_goal = self.sim.data.qvel[:6].copy()
            self.command = random.choice(self.commands)
            if self.verbose > 0:
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
            self.desired_goal = self.command

        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        self._reset_track_lst()

        return self.ob

    def do_simulation(self, action, n_frames, callback=None):
        #print(self._n_steps)
        if self._action_dim == 2:
            self._frequency = np.array([action[0]], dtype = np.float32)
            self._amplitude = 0.9 * np.array([action[1]], dtype = np.float32)
        elif self._action_dim == 4:
            self._frequency = np.array([action[0], action[2]], dtype = np.float32)
            self._amplitude = 0.9 * np.array([action[1], action[3]], dtype = np.float32)
        omega = 0.5 * 2 * np.pi * self._frequency + 1e-8
        timer_omega = omega[0]
        self.action = action
        counter = 0
        """
            modify this according to needs
        """
        reward_velocity = 0.0
        reward_energy = 0.0
        penalty = 0.0
        phase = 0.0
        if self.verbose > 0:
            print(self._n_steps)
        while(np.abs(phase) <= np.pi * self._update_action_every):
            self.joint_pos, timer_omega = self._get_joint_pos(self._amplitude, omega)
            self.last_joint_pos.pop(0)
            self.last_joint_pos.append(self.joint_pos.copy())
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
            if self.policy_type == 'MultiInputPolicy':
                """
                    modify this according to observation space
                """
                self.achieved_goal = np.concatenate([
                    velocity,
                    ang_vel
                ], -1)
            if self._is_render:
                self.render()
            if self.policy_type == 'MultiInputPolicy':
                reward_velocity += -np.linalg.norm(self.achieved_goal - self.desired_goal + 1e-9, -1)
            else:
                reward_velocity += np.linalg.norm(velocity[0] + 1e-9)
            reward_energy += -np.linalg.norm(self.sim.data.actuator_force * self.sim.data.qvel[-self._num_joints:]) + \
                -np.linalg.norm(np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
            counter += 1
            phase += timer_omega * self.dt * counter
            self._track_attr()
            self._step += 1
            if self._step % params['max_step_length'] == 0:
                break
        self._n_steps += 1
        reward_distance = np.linalg.norm(self.sim.data.qpos[:2])
        reward_velocity = np.exp(params['reward_velocity_coef'] * reward_velocity)
        reward_energy = np.exp(params['reward_energy_coef'] * reward_energy)
        reward = reward_distance + reward_velocity + reward_energy + penalty
        info = {
            'reward_velocity' : reward_velocity,
            'reward_distance' : reward_distance,
            'reward_energy' : reward_energy,
            'reward' : reward,
            'penalty' : penalty
        }

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.02 and state[2] <= 0.3
        done = not notdone

        return reward, done, info

