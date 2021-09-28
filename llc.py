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
        self.render = render
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
        if os.path.exists(
                os.path.join(self.logdir, 'tensorboard')
            ):
            shutil.rmtree(os.path.join(self.logdir, 'tensorboard'))
        os.mkdir(os.path.join(self.logdir, 'tensorboard'))
        self.__set_rl_callback()
        self.rl_model = sb3.TD3(
            'MultiInputPolicy',
            self.env,
            tensorboard_log = os.path.join(self.logdir, 'tensorboard'),
        )

    def render(self, mode='human',
           width=params['DEFAULT_SIZE'],
           height=params['DEFAULT_SIZE'],
           camera_id=None,
           camera_name=None ):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        self.omega = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.mu = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.phase = np.zeros((self.env._num_legs,), dtype = np.float32)
        self.z2 = np.concatenate([
            np.ones((self.env._num_legs,), dtype = np.float32),
            np.zeros((self.env._num_legs,), dtype = np.float32)
        ], -1)
        self.z2 = self.z1.copy()
        ob = self.env.reset()
        return ob

    def step(self, action):
        self.omega = action[:self.env._num_legs]
        self.mu = action[self.env._num_legs: 2 * self.env._num_legs]
        self.phase = action[2 * self.env._num_legs: 3 * self.env._num_legs]
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
        x = np.sqrt(mu) * np.cos(phi)
        y = np.sqrt(mu) * np.sin(phi)
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
            if self.render:
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
            if self.render:
                self.env.render()
        Z = np.stack(Z, 0)
        Z_REF  = np.stack(Z_REF, 0)
        return REWARD, [JOINT_POS, OMEGA, W, Z], [Z_REF]

    def __get_track_item(self):
        items = copy.deepcopy(self.env._track_item)
        for key in items.keys():
            items[key] = np.stack(items[key])
        items = {key: np.stack(items[key], 0) for key in items.keys()}
        return items

    def test_comparison(self, seed = 46):
        reward_ref, plot_ref, _ = self.test_env(seed = seed)
        _track_item_ref = self.__get_track_item()
        reward, plot, _ = self.test_cpg(seed = seed)
        _track_item = self.__get_track_item()

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

class Learner:
    def __init__(self,
            name = 'llc_runs',
            datapath = 'assets/out/results_v9',
            logdir = 'assets/out/results_v9',
            render = True,
        ):
        self.name = name
        self.datapath = datapath
        self.logdir = os.path.join(logdir, name)
        self.llc = LLC(
            self.name,
            self.datapath,
            self.logdir,
            render
        )
        if os.path.exists(
                os.path.join(self.logdir, 'tensorboard')
            ):
            shutil.rmtree(os.path.join(self.logdir, 'tensorboard'))
        os.mkdir(os.path.join(self.logdir, 'tensorboard'))
        self.__set_rl_callback()
        self.rl_model = sb3.TD3(
            'MultiInputPolicy',
            self.llc,
            tensorboard_log = os.path.join(self.logdir, 'tensorboard'),
        ) 

    def learn(self):
        self.rl_model.learn(
            total_timesteps = params['total_timesteps'],
            callback = self.rl_callback
        )

    def __set_rl_callback(self):
        recordcallback = CustomCallback(
            self.llc,
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
            self.llc,
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
    print(args.seed)
    learner = Learner()
    if args.task == 'test_cpg':
        learner.llc.test_cpg(seed = args.seed)
    elif args.task == 'test_env':
        learner.llc.test_env(seed = args.seed)
    elif args.task == 'test_comparison':
        learner.llc.test_comparison(seed = args.seed)
    elif args.task is None:
        learner.llc.learn()
    else:
        raise ValueError(
            'Expected one of `test_cpg` or `test_env` as task \
                input, got {}'.format(args.task))

