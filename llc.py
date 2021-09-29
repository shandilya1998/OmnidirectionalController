import numpy as np
import os
import pandas as pd
import shutil
import argparse
import copy
from constants import params
from simulations import QuadrupedV2
from oscillator import cpg_step, hopf_mod_step, hopf_simple_step
import matplotlib.pyplot as plt
import gym
import stable_baselines3 as sb3
from utils.rl_utils import CustomCallback, SaveOnBestTrainingRewardCallback
from utils.il_utils import ImitationLearning

class LLC(gym.GoalEnv, gym.utils.EzPickle):
    def __init__(
            self,
            name = 'llc_runs',
            datapath = 'assets/out/results_v9',
            logdir = 'assets/out/results_v9',
            render = True,
        ):
        gym.GoalEnv.__init__(self)
        gym.utils.EzPickle.__init__(self)
        self.track_list = params['track_list']
        self.env = QuadrupedV2(
            render = render
        )
        self.sim = self.env.sim
        self.model = self.env.model
        self._render = render
        self.dt = self.env.dt
        self.datapath = datapath
        self.logdir = os.path.join(logdir, name)
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)    
        os.mkdir(self.logdir)
        self.info = pd.read_csv(os.path.join(
            self.datapath,
            'info.csv'
        ), index_col = 0)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        low = -np.ones((12,), dtype = np.float32)
        high = np.ones((12,), dtype = np.float32)
        self._action_dim = 12
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.omega = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.w = self.omega.copy()
        self.mu = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.phase = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.z1 = np.concatenate([
            np.ones((self.env._num_legs,), dtype = np.float32),
            np.zeros((self.env._num_legs,), dtype = np.float32)
        ], -1) 
        self.z2 = self.z1.copy()

    def get_action(self):
        return self.env.cpg_action.copy()
    
    def compute_reward(self, desired_goal, achieved_goal, info):
        reward_velocity = 0.0
        if self.env.policy_type == 'MultiInputPolicy':
            reward_velocity += np.sum(np.square(
                achieved_goal - desired_goal
            ), axis = -1)
        else:
            reward_velocity += np.sum(np.square(
                achieved_goal[0] - desired_goal[0]
            ), axis = -1)
        reward_velocity = np.exp(-reward_velocity * 1e3)
        reward = 0.0
        if isinstance(info, np.ndarray):
            out = {}
            for item in info:
                for key in item.keys():
                    if key in out.keys():
                        out[key].append(item[key])
                    else:
                        out[key] = [item[key]]
            out = {
                key : np.array(out[key], dtype = np.float32) \
                    for key in out.keys()
            }
            info = out
            reward = np.concatenate([
                np.expand_dims(reward_velocity, -1),
                np.expand_dims(info['penalty'], -1),
                np.expand_dims(info['reward_energy'], -1),
                np.expand_dims(info['reward_distance'], -1),
            ], -1)
        else:
            reward = np.array([
                reward_velocity,
                info['penalty'],
                info['reward_energy'],
                info['reward_distance'],
            ], dtype = np.float32)
        reward = np.round(np.sum(reward, -1).astype(np.float32), 6)
        return reward

    def toggle_render_switch(self):
        self._render = not self._render
    
    def render(self, mode='human',
           width=params['DEFAULT_SIZE'],
           height=params['DEFAULT_SIZE'],
           camera_id=None,
           camera_name=None ):
        return self.env.render(mode = mode,
            width = width, height = height, camera_id = camera_id,
            camera_name = camera_name)

    def close(self):
        self.env.close()

    def reset(self):
        self.omega = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.mu = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.phase = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.z1 = np.concatenate([
            np.ones((self.env._num_legs,), dtype = np.float32),
            np.zeros((self.env._num_legs,), dtype = np.float32)
        ], -1)
        self.z2 = self.z1.copy()
        self.env.set_control_params(
            self.omega,
            self.mu,
            self.w,
            self.z2,
        )
        ob = self.env.reset()
        return ob

    def step(self, action):
        self.omega = action[:self.env._num_legs] * 2 * np.pi
        self.mu = action[self.env._num_legs: 2 * self.env._num_legs]
        self.phase = action[
            2 * self.env._num_legs: 3 * self.env._num_legs
        ] * np.pi
        self.env.z = self.z2
        self.env.set_control_params(
            self.omega,
            self.mu, 
            self.w,  
            self.z2,  
        )
        self.z2, self.w, self.z1 = self.cpg(
            self.omega, self.mu, self.z1, self.z2, self.phase
        )
        self.joint_pos = self.preprocess(self.z2, self.omega, self.mu)
        ob, rewards, done, info = self.env.step(self.joint_pos)
        return ob, rewards, done, info

    def cpg(self, omega, mu, z1, z2, phase):
        return cpg_step(omega, mu, z1, z2, phase, \
            self.env.C, params['degree'], self.env.dt)

    def preprocess(self, z, omega, mu):
        out = []
        num_osc = z.shape[-1] // 2
        x, y = np.split(z, 2, -1)
        phi = np.arctan2(y, x)
        x = np.sqrt(np.abs(mu)) * np.cos(phi)
        y = np.sqrt(np.abs(mu)) * np.sin(phi)
        z = np.concatenate([x, y], -1)
        for i in range(self.env._num_legs):
            direction = 1.0
            if i in [0, 3]:
                direction = 1.0
            if i in [1,2]:
                direction = -1.0
            out.append(z[i] * np.tanh(1e3 * omega[i]))
            knee = -np.maximum(-z[i + num_osc], 0)
            out.append(knee * direction)
            out.append((-0.35 * knee  + 1.3089) * direction)
        out = np.array(out, dtype = np.float32)
        return out

    def test_env(self, seed = 46):
        prng = np.random.RandomState(seed)
        index = prng.randint(low = 0, high = 6599)
        test = self.info.iloc[index]
        print(test)
        OMEGA = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega_o.npy'.format(index)
        ))
        W = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega.npy'.format(index)
        ))
        Z = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_z.npy'.format(index)
        ))
        MU = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_mu.npy'.format(index)
        ))

        omega = OMEGA[-1, :]
        mu = MU[-1, :]
        w = W[0, :]
        z = Z[0, :]
        self.env.set_control_params(
            omega,
            mu,
            w,
            z,
        )
        info = None
        REWARD = {}
        ac = self.env.action_space.sample()
        ob, reward, done, info = self.env.step(ac)
        REWARD = {key : [] for key in info.keys()}
        ob = self.env.reset()
        JOINT_POS = []
        Z = []
        for i in range(params['MAX_STEPS']):
            z, w = hopf_mod_step(
                omega,
                mu,
                z,
                self.env.C,
                params['degree'],
                self.dt
            )
            joint_pos = self.preprocess(z, omega, mu)
            ob, reward, done, info = self.env.step(joint_pos)
            JOINT_POS.append(joint_pos.copy())
            Z.append(z.copy())
            for key in info.keys():
                REWARD[key].append(info[key])
            if self._render:
                self.env.render()
        JOINT_POS = np.stack(JOINT_POS, 0)
        Z = np.stack(Z, 0)
        return REWARD, [JOINT_POS, OMEGA, W, Z], []

    def test_cpg(self, seed = 46):
        prng = np.random.RandomState(seed)
        index = prng.randint(low = 0, high = 6599)
        test = self.info.iloc[index]
        print(test)
        OMEGA = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega_o.npy'.format(index)
        ))
        MU = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_mu.npy'.format(index)
        ))
        Z = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_z.npy'.format(index)
        ))
        W = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega.npy'.format(index)
        ))
        phase = np.arctan2(
            Z[0, self.env._num_legs:], Z[0, :self.env._num_legs]        
        )
        omega = OMEGA[-1, :]
        mu = MU[-1, :]
        w = W[0, :]

        z_ref = np.concatenate([
            np.ones((self.env._num_legs,)),
            np.zeros((self.env._num_legs,))
        ], -1)
        z = np.random.random((2 * self.env._num_legs,))
        self.env.set_control_params(
            omega,
            mu, 
            w,  
            z,  
        )   
        info = None
        REWARD = {}
        ac = self.env.action_space.sample()
        ob, reward, done, info = self.env.step(ac)
        REWARD = {key : [] for key in info.keys()}
        ob = self.env.reset()
        Z = []
        Z_REF = []
        JOINT_POS = []
        for i in range(params['MAX_STEPS']):
            z, w, z_ref = self.cpg(omega, mu, z_ref, z, phase)
            Z.append(z.copy())
            Z_REF.append(z_ref.copy())
            JOINT_POS.append(self.preprocess(z, omega, mu).copy())
            ob, reward, done, info = self.env.step(JOINT_POS[-1])
            for key in info.keys():
                REWARD[key].append(info[key])
            if self._render:
                self.env.render()
        Z = np.stack(Z, 0)
        Z_REF  = np.stack(Z_REF, 0)
        return REWARD, [JOINT_POS, OMEGA, W, Z], [Z_REF]

    def get_track_item(self):
        items = copy.deepcopy(self.env._track_item)
        for key in items.keys():
            items[key] = np.stack(items[key])
        items = {key: np.stack(items[key], 0) for key in items.keys()}
        return items

    def test_comparison(self, seed = 46):
        reward_ref, plot_ref, _ = self.test_env(seed = seed)
        _track_item_ref = self.get_track_item()
        reward, plot, _ = self.test_cpg(seed = seed)
        _track_item = self.get_track_item()

        omega = _track_item_ref['omega_o'][-1]
        T = np.arange(_track_item_ref['joint_pos'].shape[0] - 1) * self.env.dt
        steps = int(2 * np.pi / (np.min(np.abs(omega)) * self.env.dt))
        fig, ax = plt.subplots(self.env._num_legs, 2, figsize = (14,7))
        for i in range(self.env._num_legs):
            ax[i][0].plot(
                T[-steps:],
                plot_ref[-1][-steps:, i],
                label = 'ref {}'.format(i),
                color = 'b',
                linestyle = '-'
            )
            ax[i][0].plot(
                T[-steps:],
                plot[-1][-steps:, i],
                label = 'cpg {}'.format(i),
                color = 'r',
                linestyle = '--'
            )
            ax[i][1].plot(
                T[-steps:],
                plot_ref[-1][-steps:, i + self.env._num_legs], 
                label = 'ref {}'.format(i),
                color = 'b',
                linestyle = '--'
            )   
            ax[i][1].plot(
                T[-steps:],
                plot[-1][-steps:, i + self.env._num_legs],
                label = 'cpg {}'.format(i),
                color = 'r',
                linestyle = '-'
            )
            ax[i][0].set_xlabel('time')
            ax[i][0].set_ylabel('real part')
            ax[i][1].set_xlabel('time')
            ax[i][1].set_ylabel('imaginary part')
        plt.show()
        plt.close()

        fig1, ax1 = plt.subplots(4, 2, figsize = (21, 14))
        joints = ['hip', 'knee1']
        for i in range(self.env._num_legs):
            for j in range(3):
                if j != 2:
                    ax1[i][j].plot(
                        T[-steps:],
                        _track_item_ref['joint_pos'][-steps:, 3 * i + j ],
                        label = 'ref {} {}'.format(joints[j], i),
                        color = 'b',
                        linestyle = '--'
                    )
                    ax1[i][j].plot(
                        T[-steps:],
                        _track_item['joint_pos'][-steps:, 3 * i + j],
                        label = 'cpg {} {}'.format(joints[j], i),
                        color = 'r',
                        linestyle = '--'
                    )
                    ax1[i][j].legend(loc = 'upper left')
                    ax1[i][j].set_xlabel('time')
                    ax1[i][j].set_ylabel('joint position')
        plt.show() 


class LocomotionGenerator:
    def __init__(self,
            env,
            mode = 'generate',
            datapath = 'assets/out/results_v9',
            logdir = 'assets/out/results_v9',
            render = False,
        ):
            self.env = env
            self.datapath = datapath
            self.logdir = logdir
            self._render = render
            self.mode = mode
            self.index = 0
            self.total_steps = 0
            if self.mode == 'generate':
                self.reset()
            elif self.mode == 'load':
                self.info = pd.read_csv(os.path.join(
                    self.datapath, 'info.csv'
                ), index_col = 0)
                self.index = len(self.info)
                self.total_steps = self.info['length'].sum(
                    axis = 0
                )

    def reset(self):
        if os.path.exists(self.datapath):
            shutil.rmtree(self.datapath)
        os.mkdir(self.datapath)
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        os.mkdir(self.logdir)
        self.info = pd.DataFrame(
            {'gait' : [], 'task' : [], 'direction' : [], 'id' : [], 'length' : []}
        )

    def done(self):
        self.info.to_csv(os.path.join(log_dir, 'info.csv'))
        print('Total Steps: {}'.format(2 * np.p.total_steps))

    def step(self,
            callback = lambda: True,
            callfreq = 1,
            env_name = 'MePed',
            run_type = 'random'
        ):
        ob = self.env.reset()
        Z = None
        W = None
        MU = None
        OMEGA = None
        z = None
        mu = np.random.uniform(
            low = params['props'][self.env.gait]['mu'][0],
            high = params['props'][self.env.gait]['mu'][1]
        )
        omega = np.random.uniform(
            low = params['props'][self.env.gait]['omega'][0],
            high = params['props'][self.env.gait]['omega'][1]
        )
        if run_type == 'constant_mu':
            if self.env.gait == 'trot':
                mu = 0.45
            else:
                mu = 0.6
        elif run_type == 'constant_omega':
            if self.env.gait == 'trot':
                omega = 4.4 / (2 * np.pi)
            else:
                omega = 2.2 / (2 * np.pi)
        z = np.concatenate([ 
            np.cos(2 * np.pi * self.env.init_gamma),
            np.sin(2 * np.pi * self.env.init_gamma)
        ], -1)
        mu = np.random.uniform(
            low = params['props'][self.env.gait]['mu'][0],
            high = params['props'][self.env.gait]['mu'][1]
        )
        omega = np.random.uniform(
            low = params['props'][self.env.gait]['omega'][0],
            high = params['props'][self.env.gait]['omega'][1]
        )
        omega = np.array(
            [omega] * self.env._num_legs, dtype = np.float32
        ) * self.env.heading_ctrl
        if self.env.task == 'straight' or self.env.task == 'rotate':
            mu = np.array([mu] * self.env._num_legs, dtype = np.float32)
        elif self.env.task == 'turn':
            mu2 = np.random.uniform(
                low = mu,
                high = props[self.env.gait]['mu']
            )
            if self.env.direction == 'left':
                mu = np.array([mu1, mu2, mu2, mu1], dtype = np.float32)
            elif env.direction == 'right':
                mu = np.array([mu2, mu1, mu1, mu2], dtype = np.float32)
        else:
            raise ValueError('Expected one of `straight`, `rotate` or \
                `turn`, got {}'.format(self.env.task))

        phase = 2 * np.pi * self.env.init_gamma

        self.env.set_control_params(
            omega,
            mu,
            w,
            z,
        ) 

        action = np.concatenate([
            omega, mu, phase
        ], -1)
            
        for i in range(params['max_episode_size']):
            if self.mode == 'generate':
                ob, reward, done, info = self.env.step(action)
                self.total_steps += 1
            elif self.mode == 'load':
                joint_pos = self.env.preprocess(Z[i, :], omega, mu)
                ob, reward, done, info = self.env.env.step(action)
            if self._render:
                self.env.render()
        if self.mode == 'generate':
            case = {
                'gait': self.env.gait,
                'task': self.env.task,
                'direction': self.env.direction,
                'id': '{}_{}'.format(env_name, self.index),
                'length': params['max_episode_size'],
                'type' : 'random'
            }
            self.info = self.info.append(case, ignore_index = True)
            for item in params['track_list']:
                with open(
                    os.path.join(
                        self.logdir, '{}_{}_{}.npy'.format(
                            env_name,
                            self.index,
                            item
                        )
                    ), 'wb'
                ) as f:
                    np.save(f, np.stack(data[item], axis = 0))
        elif self.mode == 'plot':
            raise NotImplementedError
        callback()
        self.index += 1



class Learner:
    def __init__(self,
            name = 'llc_runs',
            datapath = 'assets/out/results_v9',
            logdir = 'assets/out/results_v9',
            render = True,
        ):
        self._render = render
        self.name = name
        self.datapath = datapath
        self.logdir = os.path.join(logdir, name)
        self.llc = LLC(
            self.name,
            self.datapath,
            self.logdir,
            self._render
        )
        self.eval_llc = sb3.common.monitor.Monitor(LLC(
            self.name,
            self.datapath,
            self.logdir,
            self._render
        ))
        if os.path.exists(
                os.path.join(self.logdir, 'tensorboard')
            ):
            shutil.rmtree(os.path.join(self.logdir, 'tensorboard'))
        os.mkdir(os.path.join(self.logdir, 'tensorboard'))
        self.__set_rl_callback()
        n_actions = self.llc.action_space.sample().shape[-1]
        self.rl_model = sb3.TD3(
            'MultiInputPolicy',
            sb3.common.monitor.Monitor(self.llc),
            replay_buffer_class = sb3.her.HerReplayBuffer,
            replay_buffer_kwargs = dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
                online_sampling=True,
                max_episode_length = params['max_episode_size'],
            ),
            action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
                mean = params['OU_MEAN'] * np.ones(n_actions),
                sigma = params['OU_SIGMA'] * np.ones(n_actions)
            ),
            learning_starts = params['LEARNING_STARTS'],
            tensorboard_log = os.path.join(self.logdir, 'tensorboard'),
            train_freq = (5, "step"),
            verbose = 2,
            batch_size = params['BATCH_SIZE']
        )
        self.il_model = ImitationLearning(
            'MultiInputPolicy',
            sb3.common.monitor.Monitor(self.llc),
            tensorboard_log = os.path.join(self.logdir, 'tensorboard'),
            learning_starts = params['LEARNING_STARTS'],
            train_freq = (5, "step"),
            verbose = 2,
            batch_size = params['BATCH_SIZE']
        )
        """
        self.lg = LocomotionGenerator(
            self.llc,
            mode = 'generate',
            datapath = self.datapath,
            logdir = self.logdir,
            render = self._render
        )
        """

    def imitate(self):
        self.il_model.learn(
            total_timesteps = params['total_timesteps'],
            callback = self.rl_callback
        )

    def learn(self):
        self.rl_model.learn(
            total_timesteps = params['total_timesteps'],
            callback = self.rl_callback
        )

    def __set_rl_callback(self):
        recordcallback = CustomCallback(
            self.eval_llc,
            render_freq = params['render_freq']
        )
        if os.path.exists(os.path.join(
            self.logdir, 'checkpoints'
        )):
            shutil.rmtree(os.path.join(
                self.logdir, 'checkpoints'
            ))
        os.mkdir(os.path.join(
                self.logdir, 'checkpoints'
        ))
        checkpointcallback = sb3.common.callbacks.CheckpointCallback(
            save_freq = params['save_freq'],
            save_path = os.path.join(self.logdir, 'checkpoints'),
            name_prefix = 'rl_model'
        )
        if os.path.exists(os.path.join(
            self.logdir, 'best_model'
        )):
            shutil.rmtree(os.path.join(
                self.logdir,
                'best_model'
            ))
        os.mkdir(os.path.join(
            self.logdir,
            'best_model'
        ))
        evalcallback = sb3.common.callbacks.EvalCallback(
            self.eval_llc,
            best_model_save_path = os.path.join(
                self.logdir, 'best_model'
            ),
            eval_freq = params['eval_freq'],
            log_path = self.logdir
        )
        self.rl_callback = sb3.common.callbacks.CallbackList([
            checkpointcallback,
            recordcallback,
            evalcallback,
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--seed',
        nargs='?', type = int, const = 1, default = 46,
        help = 'Seed for random number generator'
    )
    args = parser.parse_args()
    learner = Learner()
    if args.task == 'test_cpg':
        learner.llc.toggle_render_switch()
        learner.llc.test_cpg(seed = args.seed)
    elif args.task == 'test_env':
        learner.llc.toggle_render_switch()
        learner.llc.test_env(seed = args.seed)
    elif args.task == 'test_comparison':
        learner.llc.toggle_render_switch()
        learner.llc.test_comparison(seed = args.seed)
    elif args.task == 'train_il':
        learner.imitate()
    elif args.task == 'train_rl':
        learner.learn()
    else:
        raise ValueError(
            'Expected one of `test_cpg`, `test_env`, `test_comparison` or \
                `None` as task input, got {}'.format(args.task))

