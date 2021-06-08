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

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class Quadruped(gym.Env, utils.EzPickle):
    def __init__(self,
                 model_path,
                 frame_skip = 5,
                 render = True,
                 gait = 'ls_crawl',
                 task = 'straight',
                 direction = 'forward',
                 policy_type = 'MultiInputPolicy',
                 stairs = False):
        gym.Env.__init__(self)
        utils.EzPickle.__init__(self)

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
        assert self.gait in [
            'ds_crawl',
            'ls_crawl',
            'trot',
            'pace',
            'bound',
            'transverse_gallop',
            'rotary_gallop'
        ]
        assert self.task in [
            'rotate',
            'turn',
            'straight'
        ]
        assert self.direction in [
            'forward',
            'backward',
            'left',
            'right'
        ]
        assert not (self.gait not in ['ds_crawl', 'ls_crawl'] and self.task == 'rotate')
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

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._is_stairs = stairs
        self._is_render = render
        self._num_joints = self.init_qpos.shape[-1] - 7
        self._num_legs = 4

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()
        self.action = np.zeros(self._action_dim)
        self._distance_limit = float("inf")

        self.init_qpos = np.concatenate([
            self._last_base_position, # base position
            np.zeros((4,), dtype = np.float32), # base angular position (quaternion)
            params['INIT_JOINT_POS'] # joint angular position
        ], -1)

        self.init_qvel = np.concatenate([
            np.zeros((3,), dtype = np.float32), # base translational velocity
            np.zeros((4,), dtype = np.float32), # base angular velocity (euler)
            np.zeros(shape = params['INIT_JOINT_POS'].shape, dtype = np.float32) # joint angular velocity
        ], -1)

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
                    yaw_rate = np.arange(-0.1, 0.001, 0.001).tolist()
                elif self.direction == 'right':
                    yaw_rate = np.arange(0.001, 0.1, 0.001).tolist()
            elif self.task == 'straight':
                if self.direction == 'left':
                    yvel = np.arange(-0.1, 0.001, 0.001).tolist()
                elif self.direction == 'right':
                    yvel = np.arange(0.001, 0.1, 0.001).tolist()
                elif self.direction == 'forward':
                    xvel = np.arange(0.001, 0.1, 0.001).tolist()
                elif self.direction == 'backward':
                    xvel = np.arange(-0.1, 0.001, 0.001).tolist()
            else:
                raise ValueError
            if self.task == 'turn':
                xvel = np.arange(1.0, 0.001, 0.001).tolist()

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
        self.last_joint_pos = [self.init_qpos[-self._num_joints:]]
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
            low = np.zeros((2,), dtype = np.float32)
            high = np.ones((2,), dtype = np.float32)
            self._action_dim = 2
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            return self.action_space
        else:
            self._action_dim = 4
            low = np.zeros((4,), dtype = np.float32)
            high = np.ones((4,), dtype = np.float32)
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
            self.init_gamma = [0, 0.5, 0.75, 0.25]
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
            if self.direction == 'left':
                temp = self.init_gamma.pop(0)
                self.init_gamma.append(temp)
            elif self.direction == 'right':
                temp = self.init_gamma.pop(-1)
                self.init_gamma.inset(0, temp)
            elif self.direction == 'backward':
                self.init_gamma = self.init[-2:] + self.init_gamma[:2]
        self.init_gamma = np.array(self.init_gamma, dtype = np.float32) * np.pi
        return self.init_gamma

    def reset(self):
        self._last_base_position = [0, 0, params['INIT_HEIGHT']]
        self.gamma = self.init_gamma.copy()
        self.sim.reset()
        self.ob = self.reset_model()
        if self.policy_type == 'MultiInputPolicy':
                """
                    modify this according to observation space
                """
                self.achieved_goal = self.sim.data.qvel[:6].copy()
                self.command = random.choice(self.commands)
                print('[Quadruped] Command is `{}` with gait `{}` in task `{}` and direction `{}`'.format(self.command, self.gait, self.task, self.direction))
                self.desired_goal = self.command
        return {
            'observation' : self.ob,
            'desired_goal' : self.desired_goal,
            'achieved_goal' : self.achieved_goal
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size = self.model.nq, low = -0.1, high = 0.1)
        qvel = self.init_qvel + self.np_random.randn(size = self.model.nv, low = -0.1, high = 0.1)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
            modify this according to observation space
        """
        ob = {}
        if self.policy_type == 'MultiInputPolicy':
            ob = {
                'observation' : np.concatenate(self.last_joint_pos, -1),
                'desired_goal' : self.desired_goal,
                'achieved_goal' : self.achieved_goal
            }
        else:
            ob = np.concatentate(self.last_joint_pos, -1)

        return ob

    def step(self, action, callback=None):
        action = np.clip(action, -1, 1) # modify this according to appropriate bounds
        reward, done, info = self.do_simulation(action, n_frames = self._frame_skip)
        ob = self._get_obs()
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _get_joint_pos(self, amplitude, gamma, counter, current_omega):
        """
            modify this according to joint value limits in xml file and leg construction
        """
        out = []
        direction = 1.0
        max_amplitude = 1.0471
        for joint in range(self._num_joints):
            max_amplitude = 1.0471
            bias = 0.0
            amp = amplitude[0]
            if joint == 0 or joint == 9:
                direction = 1.0
            elif joint == 3 or joint == 6:
                direction =  -1.0
            elif joint in [1, 10]:
                if gamma[joint % 3] > np.pi / 2 and gamma[joint % 3] < 3 * np.pi / 2:
                    direction = 1.0
                else:
                    direction = 0.0
            elif joint in [4, 7]:
                if gamma[joint % 3] > np.pi / 2 and gamma[joint % 3] < 3 * np.pi / 2:
                    direction = -1.0
                else:
                    direction = 0.0
            """
            if joint in [2, 5, 8, 11]:
                max_amplitude = 1.7155
                bias = 1.2217
            """
            if self._action_dim == 4:
                if joint in [3, 4, 6, 7]:
                    amp = amplitude[1]
            if joint in [2, 5, 8, 11]:
                """
                    modify this according to leg construction
                """
                knee_pos = out[-1]
                out.append(self._leg_kinematics(knee_pos))
            else:
                out.append(amp * np.sin(gamma[joint % 3]) * max_amplitude * direction + bias)
        gamma += current_omega * counter * self.dt
        gamma = gamma % (2 * np.pi)
        #print(out)
        return np.array(out, dtype = np.float32), gamma

    def _get_current_omega(self, omega_st, omega_sw, gamma):
        current_omega = omega_st
        for leg in range(self._num_legs):
            if gamma[leg] > np.pi / 2 and gamma[leg] < 3 * np.pi / 2:
                current_omega = omega_sw
            else:
                current_omega = omega_st
        return current_omega

    def _leg_kinematics(self, knee_pos):
        """
            modify this according to leg construction
        """
        t = np.sqrt(self.p ** 2 + self.q ** 2 + self.r ** 2 - 2 * np.sqrt(self.p ** 2 + self.q ** 2) * self.r * np.cos(knee_pos))
        print((self.c ** 2 + t ** 2 - self.s ** 2) / (2 * self.c * t))
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
        alpha = np.arctan((Cy - Ay) / (Cx - Ax)) - np.pi
        return alpha

    def _track_attr(self):
        """
            modify this according to need
        """
        raise NotImplementedError


    def do_simulation(self, action, n_frames, callback=None):
        if self._action_dim == 2:
            self._frequency = np.array([action[0]], dtype = np.float32)
            self._amplitude = np.array([action[1]], dtype = np.float32)
        elif self._action_dim == 4:
            self._frequency = np.array([action[0], action[2]], dtype = np.float32)
            self._amplitude = np.array([action[1], action[3]], dtype = np.float32)
        omega = 2 * np.pi * self._frequency + 1e-8
        omega_st = omega / (self.beta + 1e-8)
        omega_sw = omega / (1 - self.beta + 1e-8)
        current_omega = self._get_current_omega(omega_st, omega_sw, self.gamma)
        self.action = action
        counter = 0
        """
            modify this according to needs
        """
        reward_velocity = 0.0
        reward_energy = 0.0
        penalty = 0.0
        phase = 0.0
        while(np.abs(phase) <= np.pi * self._update_action_every):
            self.joint_pos, self.gamma = self._get_joint_pos(self._amplitude, self.gamma, counter, current_omega)
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
                reward_velocity += np.linalg.norm(self.achieved_goal - self.desired_goal, -1)
            else:
                reward_velocity += np.linalg.norm(velocity[0])
            reward_energy += -np.linalg.norm(self.sim.data.actuator_force * self.sim.data.qvel[-self._num_joints:]) + \
                -np.linalg.norm(np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
            current_omega = self._get_current_omega(omega_st, omega_sw, self.gamma, )
            counter += 1
            phase += current_omega.min() * self.dt * counter
        self._n_steps += 1
        reward = reward_velocity + reward_energy * params['reward_energy_coef'] + penalty
        info = {
            'reward_velocity' : reward_velocity,
            'reward_energy' : reward_energy * params['reward_energy_coef'],
            'reward' : reward,
            'penalty' : penalty
        }

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.02 and state[2] <= 0.3
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
