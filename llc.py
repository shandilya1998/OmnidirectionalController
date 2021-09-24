import numpy as np
import os
import pandas as pd
import shutil
from constants import params
from simulations import QuadrupedV2
from oscillator import cpg_step

class LLC:
    def __init__(
        self,
        datapath = 'assets/out/results_v9',
        logdir = 'assets/out/results_v9',
        render = False,
        ):
        self.env = QuadrupedV2(
            render = render
        )
        self.dt = self.env.dt
        self.datapath = datapath
        self.logdir = os.path.join(logdir, 'llc_tests')
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)    
        os.mkdir(self.logdir)
        self.info = pd.read_csv(os.path.join(
            self.datapath,
            'info.csv'
        ), index_col = 0)

    def cpg(self, omega, mu, z1, z2, phase):
        return cpg_step(omega, mu, z1, z2, phase, \
            self.env.C, params['degree'], self.dt)

    def preprocess(self, z, omega):
        num_osc = z.shape[-1] // 2
        out = []
        for i in range(self._num_legs):
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

    def test_env(self):
        index = np.random.randint(low = 0, high = 6599)
        test = self.info.iloc[index]
        print(test)
        JOINT_POS = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_joint_pos.npy'.format(index)
        ))
        OMEGA = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega_o.npy'.format(index)
        ))
        W = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega.npy'.format(index)
        ))

        omega = OMEGA[-1, :]
        w = W[0, :]
        mu = np.ones((omega.shape[-1]))

        z = np.concatenate([
            np.ones((self.env._num_legs,)),
            np.zeros((self.env._num_legs,))
        ], -1)
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
        for i in range(JOINT_POS.shape[0]):
            ob, reward, done, info = self.env.step(JOINT_POS[i, :])
            for key in info.keys():
                REWARD[key].append(info[key])
            self.env.render()
        return REWARD

    def test_cpg(self):
        index = np.random.randint(low = 0, high = 6599)
        test = self.info.iloc[index]
        print(test)
        OMEGA = np.load(os.path.join(
            self.datapath,
            'Quadruped_{}_omega_o.npy'.format(index)
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
        w = W[0, :]
        mu = np.ones((omega.shape[-1]))

        z = np.concatenate([
            np.ones((self.env._num_legs,)),
            np.zeros((self.env._num_legs,))
        ], -1)
        z_ref = z.copy()
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
        _Z = []
        _Z_REF = []  
        for i in range(JOINT_POS.shape[0]):
            z, w, z_ref = self.cpg(omega, mu, z1, z2, phase)
            _Z.append(z.copy())
            _Z_REF.append(z_ref.copy())
            ob, reward, done, info = self.env.step(self.preprocess(z, omega))
            for key in info.keys():
                REWARD[key].append(info[key])
            self.env.render()
        return REWARD, _Z, _Z_REF

if __name__ == '__main__':
    llc = LLC()
    llc.test()
