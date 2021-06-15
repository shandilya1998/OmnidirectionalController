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
from utils import convert_observation_to_space
from oscillator import hopf_step, _get_polynomial_coef

class Quadruped(gym.GoalEnv, utils.EzPickle):
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
        gym.Env.__init__(self)
        utils.EzPickle.__init__(self)
        self._track_lst = track_lst
        self._track_item = {key : [] for key in self._track_lst}
        self._step = 0
        self.verbose = 0
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.getcwd(), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self._frame_skip = frame_skip
        self.gait = gait
        self.task = task
        self.direction = direction
        self.policy_type = policy_type
        assert self.gait in params['gait_list']
        assert self.task in params['task_list']
        assert self.direction in params['direction_list']
        assert not (self.gait not in ['ds_crawl', 'ls_crawl', 'trot'] and self.task == 'rotate')
        assert not (self.direction not in ['left', 'right'] and self.task == 'rotate')
        assert not (self.direction not in ['left', 'right'] and self.task == 'turn')
        self._n_steps = 0
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self._update_action_every = 1.
        self._frequency = 2.8
        self.model.opt.timestep = params['dt']
        self._last_base_position = [0, 0, params['INIT_HEIGHT']]

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = np.concatenate([
            self._last_base_position, # base position
            np.zeros((4,), dtype = np.float32), # base angular position (quaternion)
            params['INIT_JOINT_POS'] # joint angular position
        ], -1)

        self.init_qvel = np.concatenate([
            np.zeros((3,), dtype = np.float32), # base translational velocity
            np.zeros((3,), dtype = np.float32), # base angular velocity (euler)
            np.zeros(shape = params['INIT_JOINT_POS'].shape, dtype = np.float32) # joint angular velocity
        ], -1)

        self._is_stairs = stairs
        self._is_render = render
        self._num_joints = self.init_qpos.shape[-1] - 7
        self._num_legs = 4
        self.joint_pos = self.sim.data.qpos[-self._num_joints:]

        self._set_action_space()
        action = self.action_space.sample()
        self.action = np.zeros(self._action_dim)
        self._track_attr()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()
        self._distance_limit = float("inf")

        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0


    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _create_command_lst(self):
        self.commands = []
        xvel = [0.0]
        yvel = [0.0]
        zvel = [0.0]
        roll_rate = [0.0]
        pitch_rate = [0.0]
        yaw_rate = [0.0]
        if self.gait in [
            'ds_crawl',
            'ls_crawl',
            'trot',
            'pace',
            'bound',
            'transverse_gallop',
            'rotary_gallop',
        ]:
            if self.task == 'rotate' or self.task == 'turn':
                if self.direction == 'left':
                    yaw_rate = np.arange(-0.05, 0.001, 0.001).tolist()
                elif self.direction == 'right':
                    yaw_rate = np.arange(0.001, 0.05, 0.001).tolist()
            elif self.task == 'straight':
                if self.direction == 'left':
                    yvel = np.arange(-0.05, 0.001, 0.001).tolist()
                elif self.direction == 'right':
                    yvel = np.arange(0.001, 0.05, 0.001).tolist()
                elif self.direction == 'forward':
                    xvel = np.arange(0.001, 0.05, 0.001).tolist()
                elif self.direction == 'backward':
                    xvel = np.arange(-0.05, 0.001, 0.001).tolist()
            else:
                raise ValueError

        for _x in xvel:
            for _y in yvel:
                for _z in zvel:
                    for _roll in roll_rate:
                        for _pitch in pitch_rate:
                            for _yaw in yaw_rate:
                                self.commands.append(np.array([
                                    _x, _y, _z, _roll, _pitch, _yaw
                                ], dtype = np.float32))
        return self.commands

    @property
    def dt(self):
        return self.model.opt.timestep * self._frame_skip

    def _set_leg_params(self):
        """
            modify this according to leg construction
        """
        self.p = 0.01600
        self.q = 0.00000
        self.r = 0.02000
        self.c = 0.01811
        self.u = 0.00000
        self.v = 0.00000
        self.e = -0.06000
        self.h = -0.02820
        self.s = 0.02200

    def _set_action_space(self):
        self.init_b = np.concatenate([self.joint_pos, self.sim.data.sensordata.copy()], -1)
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
            low = np.array([0, 0], dtype = np.float32)
            high = np.array([1, 1], dtype = np.float32)
            self._action_dim = 2
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            return self.action_space
        else:
            self._action_dim = 4
            low = np.array([0, 0, 0, 0], dtype = np.float32)
            high = np.array([1, 1, 1, 1], dtype = np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            return self.action_space

    def _set_beta(self):
        if self.gait == 'ds_crawl' or self.gait == 'ls_crawl':
            self.beta = 0.75
        elif self.gait == 'trot' or self.gait == 'pace' or self.gait == 'turn':
            self.beta = 0.5
        elif self.gait == 'bound' or self.gait == 'transverse_gallop' or self.gait == 'rotary_gallop':
            self.beta = 0.25

    def _set_init_gamma(self):
        if self.gait == 'ds_crawl':
            self.init_gamma = [0.0, 0.5, 0.75, 0.25]
        elif self.gait == 'ls_crawl':
            self.init_gamma = [0, 0.5, 0.25, 0.75]
        elif self.gait == 'trot':
            self.init_gamma = [0, 0.5, 0, 0.5]
        elif self.gait == 'pace':
            self.init_gamma = [0, 0.5, 0.5, 0]
        elif self.gait == 'bound':
            self.init_gamma = [0, 0, 0.5, 0.5]
        elif self.gait == 'transverse_gallop':
            self.init_gamma = [0, 0.1, 0.6, 0.5]
        elif self.gait == 'rotary_gallop':
            self.init_gamma = [0, 0.1, 0.5, 0.6]
        if self.task == 'straight':
            if self.direction == 'forward':
                self.heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
            if self.direction == 'backward':
                self.init_gamma = self.init_gamma[-2:] + self.init_gamma[:2]
                self.heading_ctrl = np.array([-1.0, 1.0, 1.0, -1.0], dtype = np.float32)
            if self.direction == 'left':
                self.init_gamma = self.init_gamma[1:] + self.init_gamma[:1]
                self.heading_ctrl = np.array([1.0, 1.0, -1.0, -1.0], dtype = np.float32)
            if self.direction == 'right':
                self.init_gamma = self.init_gamma[-1:] + self.init_gamma[:-1]
                self.heading_ctrl = np.array([-1.0, -1.0, 1.0, 1.0], dtype = np.float32)
        if self.task == 'rotate':
            if self.direction == 'left':
                self.init_gamma = [0.75, 0.5, 0.25, 0.0]
                self.heading_ctrl = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32)
            if self.direction == 'right':
                self.init_gamma = [0, 0.25, 0.5, 0.75]
                self.heading_ctrl = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32)
        if self.task == 'turn':
            self.heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
        self.init_gamma = np.array(self.init_gamma, dtype = np.float32)
        #self.z = np.concatenate([np.cos((self.init_gamma + params['offset']) * np.pi * 2), np.sin((self.init_gamma + params['offset']) * np.pi * 2)], -1)
        #print(self.init_gamma)
        self.z = self._get_z()
        self.heading_ctrl *= 1.0
        self.C = np.load('assets/out/plots/coef.npy')
        return self.init_gamma

    def _get_z(self):
        out = []
        phi = []
        for i in range(4):
            if params['offset'][i] + self.init_gamma[i] == 1.0:
                phi.append(0.0)
            else:
                phi.append(params['offset'][i] + self.init_gamma[i])
        phi = np.array(phi, dtype = np.float32)
        print(np.cos(phi * 2 * np.pi))
        phi = phi + np.cos(phi * 2 * np.pi) * 3 * (1 - self.beta) / 8
        return np.concatenate([np.cos(phi * 2 * np.pi), np.sin(phi * 2 * np.pi)], -1)

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
            self.command = random.choice(self.commands)
            if self.verbose > 0:
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
            self.desired_goal = self.command

        if len(self._track_lst) > 0 and self.verbose > 0:
            for item in self._track_lst:
                with open(os.path.join('assets', 'episode','ant_{}.npy'.format(item)), 'wb') as f:
                    np.save(f, np.stack(self._track_item[item], axis = 0))
        self._reset_track_lst()
        self._track_attr()
        return self.ob

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

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

    def step(self, action, callback=None):
        action = np.clip(action, -1, 1) # modify this according to appropriate bounds
        reward, done, info = self.do_simulation(action, n_frames = self._frame_skip)
        ob = self._get_obs()
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _leg_kinematics(self, knee_pos):
        """
            modify this according to leg construction
        """
        knee_pos = np.abs(knee_pos)
        sign = np.sign(knee_pos)
        t = np.sqrt(self.p ** 2 + self.q ** 2 + self.r ** 2 - 2 * np.sqrt(self.p ** 2 + self.q ** 2) * self.r * np.cos(knee_pos))
        phi = np.arccos((self.c ** 2 + t ** 2 - self.s **2) / (2 * self.c * t))
        delta = np.arcsin((self.c * np.sin(phi)) / self.s)
        beta = np.arcsin((self.r * np.sin(knee_pos)) / t)
        epsilon = np.pi - (delta + beta)
        Ax = self.r * np.cos(knee_pos) + self.u
        Ay = self.r * np.sin(knee_pos) + self.v
        Bx = self.s * np.cos(epsilon) + self.u + self.p
        By = self.s * np.cos(epsilon) + self.v + self.q
        Cx = Ax + ((Bx - Ax) * self.e + (Ay - By) * self.h) / self.c
        Cy = Ay + ((By - Ay) * self.e + (Ax - Bx) * self.h) / self.c
        alpha = np.arctan((Cy - Ay) / (Cx - Ax))
        return alpha * sign

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

    def _get_track_item(self, item):
        return self._track_item[item].copy()

    def _reset_track_lst(self):
        """
             modify this according to need
        """
        del self._track_item
        self._track_item = {key : [] for key in self._track_lst}
        return self._track_item

    def _compute_joint_pos(self, t, T, theta_h, theta_k, beta, direction):
        """
            modify according to need
        """
        t = t % T
        theta1 = 0.0
        theta2 = 0.0
        theta3 = 0.0
        omega = 0.0
        if 0 <= t <= beta * T/2:
            theta1 = theta_h * np.sin(
                np.pi * t / (
                    T * beta
                ) + np.pi
            )
            omega = 2 * np.pi / (T * beta)
            theta2 = 0.0
        elif T * beta / 2 < t < T * (2 - beta) / 2:
            theta1 = theta_h * np.sin(
                np.pi * t / (
                    T * (1 - beta)
                ) + np.pi * (3 - 4 * beta)/(2 * (1 - beta))
            )
            theta2 = theta_k * np.sin(
                np.pi * t/(
                    T * (1 - beta)
                ) - np.pi * beta/(2 * (1 - beta))
            )
            omega = 2 * np.pi / (T * (1 - beta))
        elif T * (2 - beta) / 2 <= t < T:
            theta1 = theta_h * np.sin(
                np.pi * t / (
                    T * beta
                ) + np.pi * (beta - 1) / beta
            )
            theta2 = 0.0
            omega = 2 * np.pi / (T * beta)
        #theta3 = self._leg_kinematics(theta2)  + direction * 0.75
        theta3 = 0.5 * theta2
        return [direction * theta1, -direction * theta2, direction * (theta3 + 1.3089)], omega

    def _get_joint_pos(self, amplitude, omega):
        """
            modify this according to joint value limits in xml file and leg construction
        """
        out = []
        amp = []
        omg = []
        amp.append(amplitude[0])
        omg.append(omega[0])
        timer_omega = 0.0
        omegas = []
        if self.task == 'turn':
            amp.append(amplitude[1])
            omg.append(omega[1])
        else:
            amp.append(amplitude[0])
            omg.append(omega[0])
        for leg in range(self._num_legs):
            if leg in [0, 3]:
                T = 2 * np.pi / omg[0]
                ac, timer_omega = self._compute_joint_pos(self._step * self.dt + self.gamma[leg] * T, T, amp[0] * 1.0471, amp[0] * 1.0471, self.beta, 1.0)
                omegas.append(timer_omega)
                out.extend(ac)
            elif leg in [1,2]:
                T = 2 * np.pi / omg[1]
                ac, timer_omega = self._compute_joint_pos(self._step * self.dt + self.gamma[leg] * T, T, amp[1] * 1.0471, amp[1] * 1.0471, self.beta, -1.0)
                out.extend(ac)
                omegas.append(timer_omega)
        out = np.array(out, dtype = np.float32)
        if self.task == 'straight':
            if self.direction == 'backward':
                #out = -out[np.array([6, 7 ,8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype = np.int32)]
                out[np.array([0, 3, 6, 9], dtype = np.int32)] *= -1.0
            elif self.direction == 'left':
                out[np.array([3, 9])] *= -1.0
            elif self.direction == 'right':
                out[np.array([0, 6])] *= -1.0
        elif self.task == 'rotate':
            if self.direction == 'left':
                out[np.array([3, 6], dtype = np.int32)] *= -1.0
            elif self.direction == 'right':
                out[np.array([0, 9], dtype = np.int32)] *= -1.0
        return np.array(out, dtype = np.float32), timer_omega

    def _get_joint_pos_v2(self, mu, omega):
        out = []
        amp = []
        omg = []
        if self._action_dim == 2:
            amp.extend([mu[0]] * 4)
            omg.extend([omega[0]]* 4)
        elif self._action_dim == 4:
            amp.extend([mu[0], mu[1], mu[1], mu[0]])
            omg.extend([omega[0], omega[1], omega[1], omega[0]])
        time_omega = 0.0
        self.mu = np.array(amp, dtype = np.float32)
        self.omega = np.array(omg, dtype = np.float32) * self.heading_ctrl
        self.z, w = hopf_step(self.omega, self.mu, self.z, self.C, params['degree'])
        out = []
        for i in range(self._num_legs):
            direction = 1.0
            if i in [0, 3]:
                direction = 1.0
            if i in [1,2]:
                direction = -1.0
            out.append(self.z[i] * np.tanh(1e3 * self.omega[i]))
            knee = -np.maximum(-self.z[self._num_legs + i], 0)
            out.append(knee * direction)
            out.append((-0.35 * knee  + 1.3089) * direction)
        out = np.array(out, dtype = np.float32)
        return out, w.max()

    def do_simulation(self, action, n_frames, callback=None):
        #print(self._n_steps)
        if self._action_dim == 2:
            self._frequency = np.array([action[0]], dtype = np.float32)
            self._amplitude = np.array([action[1]], dtype = np.float32)
        elif self._action_dim == 4:
            self._frequency = np.array([action[0], action[2]], dtype = np.float32)
            self._amplitude = np.array([action[1], action[3]], dtype = np.float32)
        omega = 2 * np.pi * self._frequency
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
            if params['version'] == 0:
                self.joint_pos, timer_omega = self._get_joint_pos(self._amplitude, omega)
            elif params['version'] == 1:
                self.joint_pos, timer_omega = self._get_joint_pos_v2(self._amplitude, omega)
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
            and state[2] >= 0.02 and state[2] <= 0.5
        done = not notdone

        return reward, done, info

    def render(self,
               mode='human',
               width=params['DEFAULT_SIZE'],
               height=params['DEFAULT_SIZE'],
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
