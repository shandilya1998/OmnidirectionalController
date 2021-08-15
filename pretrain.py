# -*- coding: utf-8 -*-
import os
import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3 import TD3, HerReplayBuffer, PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import torch
from torchsummary import summary
from constants import params
from gym import envs
import imp
import sys
from utils.plot_utils import *
from utils import *
from torch.fft import fft

np.seterr('ignore')

import sys
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'cuda'

def __import__(name, globals=None, locals=None, fromlist=None):
    # Fast path: see if the module has already been imported.
    try:
        return sys.modules[name]
    except KeyError:
        pass

    # If any of the following calls raises an exception,
    # there's a problem we can't handle -- let the caller handle it.

    fp, pathname, description = imp.find_module(name)

    try:
        return imp.load_module(name, fp, pathname, description)
    finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()


class Learner:
    def __init__(self, out_path, env, model_class, logger):
        self.log_dir = out_path
        self.env = env
        self.logger = logger
        self.env
        self.env.ref_info['start'] = np.zeros((len(self.env.ref_info),), dtype = np.int32)
        self.env.ref_info.dropna(inplace = True)
        self._epoch = 0
        self._step = 0
        self._model = model_class()
        self._model.to(DEVICE)
        self._optim  = torch.optim.Adam(
            self._model.parameters(),
            lr = params['LEARNING_RATE']
        )
        self._n_step = 0
        self._n_epoch = 0
        self._mse = torch.nn.functional.mse_loss
        self._waitlist_samples = self.env.ref_info.sample(params['BATCH_SIZE']).reset_index(drop = True).copy()
        self.init_osc = np.concatenate([np.ones((params['units_osc'],), dtype = np.float32), np.zeros((params['units_osc'],), dtype = np.float32)], -1)
        #self._get_data()


    def _get_remaining_steps(self, remaining, index, arrays, items):
        sample = self.env.ref_info.sample().copy().reset_index()
        for col in self.env.ref_info.columns:
                self._waitlist_samples.loc[index, col] = sample.loc[0, col]
        if self._waitlist_samples['length'][index] - self._waitlist_samples['start'][index] > remaining:
            f = self._waitlist_samples['id'][index]
            start = self._waitlist_samples['start'][index]
            for item in items:
                remaining_arr = np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, item)))[start:remaining, :]
                arrays[item] = np.concatenate([arrays[item], remaining_arr.copy()], 0)
            self._waitlist_samples.loc[index, 'start'] = self._waitlist_samples.loc[index, 'start'] + remaining
            return arrays
        else:
            f = self._waitlist_samples['id'][index]
            start = self._waitlist_samples['start'][index]
            for item in items:
                remaining_arr = np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, item)))[start:, :]
                arrays[item] = np.concatenate([arrays[item], remaining_arr.copy()], 0)
            remaining = remaining - (self._waitlist_samples['length'][index] - start)
            if remaining == 0:
                sample = self.env.ref_info.sample().copy().reset_index()
                for col in self.env.ref_info.columns:
                    self._waitlist_samples.loc[index, col] = sample.loc[0, col]
                return arrays
            else:
                return self._get_remaining_steps(remaining, index, arrays, items)

    def _get_data(self):
        self._current_samples = self._waitlist_samples.copy()
        files = self._current_samples['id'].values.tolist()
        starts = self._current_samples['start'].values.tolist()
        steps = params['window_length']
        achieved_goal = [
            np.load(
                os.path.join(
                    params['ref_path'],
                    '{}_{}.npy'.format(
                        f,
                        'achieved_goal'
                    )
                )
            )[start:, :] for f,start in zip(files,starts)
        ]
        ob = [
            np.load(
                os.path.join(
                    params['ref_path'],
                    '{}_{}.npy'.format(
                        f,
                        'observation'
                    )
                )
            )[start:, :] for f,start in zip(files,starts)
        ]
        y = [
            np.load(
                os.path.join(
                    params['ref_path'],
                    '{}_{}.npy'.format(
                        f,
                        'joint_pos'
                    )
                )
            )[start:, :] for f, start in zip(files, starts)
        ]
        for i in range(len(self._current_samples)):
            if self._current_samples['length'][i] - starts[i] < steps:
                remaining = steps - (self._current_samples['length'][i] - starts[i])
                arrays = self._get_remaining_steps(remaining, i, {
                    'achieved_goal' : achieved_goal[i],
                    'observation' : ob[i],
                    'joint_pos' : y[i]
                }, ['achieved_goal', 'observation', 'joint_pos'])
                achieved_goal[i] = arrays['achieved_goal']
                ob[i] = arrays['observation']
                y[i] = arrays['joint_pos']
            elif self._current_samples['length'][i] - starts[i] == steps:
                sample = self.env.ref_info.sample().copy().reset_index()
                for col in self.env.ref_info.columns:
                    self._waitlist_samples.loc[i, col] = sample.loc[0, col]
            else:
                achieved_goal[i] = achieved_goal[i][:steps, :]
                ob[i] = ob[i][:steps, :]
                y[i] = y[i][:steps, :]
                self._current_samples.loc[i, 'start'] = self._current_samples.loc[i, 'start'] + int(steps)
                self._waitlist_samples.loc[i, 'start'] = self._waitlist_samples.loc[i, 'start'] + int(steps)
        out = []
        count = 0
        for i, item in enumerate(y):
            if item.shape[0] != 2500:
                print(i, item.shape)
        achieved_goal = np.stack(achieved_goal, 0)
        y = np.stack(y, 0)
        out.append(achieved_goal[:, 0, :])
        for i in range(1, achieved_goal.shape[1]):
            if count > 500:
                count += 1
            out.append(np.mean(achieved_goal[:, count:count+i, :], axis = 1))
        desired_goal = np.nan_to_num(np.stack(out, axis = 1))
        x = np.concatenate([desired_goal, achieved_goal, ob], -1)
        x = to_tensor(x)
        y = to_tensor(y[:, :, np.array([0, 1, 3, 4, 6, 7, 9, 10])])
        return x, y, steps


    def _save(self, experiment):
        torch.save(self._model, os.path.join(self.log_dir, 'exp{}'.format(experiment),'controller.pth'))

    def _eval(self):
        ob = self.env.reset()
        total_reward = 0.0
        done = False
        while not done:
            ac = self._model(torch.cat([
                to_tensor(np.expand_dims(ob['desired_goal'], 0)),
                to_tensor(np.expand_dims(ob['achieved_goal'], 0)),
                to_tensor(np.expand_dims(ob['observation'], 0))
            ], -1))
            ob, reward, done, info = env.step(ac[0].detach().cpu().numpy())
            total_reward += reward
        return total_reward

    def learn(self, experiment):
        print('Start Pretraining.')
        self._ep = 0
        while self._ep < params['n_episodes']:
            loss = self._pretrain(experiment)
            print('Episode {} Loss {:.6f}'.format(self._ep, loss))
            self.logger.add_scalar('Train/Episode Loss', loss, self._ep)
            self._ep += 1
        print('Pretraining Done.')

    def _pretrain_step(self, x, y):
        loss = 0.0
        self._model.zero_grad()
        y_pred = self._model(x)
        FFT = fft(y, n = params['h'], dim = 1)
        y = torch.cat([FFT.real, FFT.imag], 1)
        y = torch.transpose(y, 1, 2)
        loss += torch.nn.functional.mse_loss(y_pred, y)
        self._step += params['stride']
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        self.logger.add_scalar('Train/Loss', loss.detach().cpu().numpy(), self._n_step)
        self._n_step += 1
        return loss.detach().cpu().numpy()

    def _pretrain_epoch(self, x, y, steps):
        epoch_loss = 0.0
        self._step = 0
        while self._step < steps - params['h']:
            epoch_loss += self._pretrain_step(torch.squeeze(x[:, self._step: self._step + params['h'], :], 1), y[:, self._step: self._step + params['h'], :])
        return epoch_loss

    def _pretrain(self, experiment):
        """
            modify according to need
        """
        steps = params['min_epoch_size']
        ep_loss = 0.0
        self._epoch = 0
        while self._epoch < params['n_epochs']:
            x, y, steps = self._get_data()
            if steps > params['h']:
                while steps < params['h'] + params['stride']:
                    x_, y_, steps_ = self.get_data()
                    x = torch.cat([x, x_], 1)
                    y = torch,cat([y, y_], 1)
                    steps += steps_
            epoch_loss = self._pretrain_epoch(x, y, steps)
            self._epoch += 1
            if steps > 0:
                epoch_loss = epoch_loss / steps
            ep_loss += epoch_loss
            self.logger.add_scalar('Train/Epoch Loss', epoch_loss, self._n_epoch)
            self._n_epoch += 1
            if (self._ep + 1 ) * self._epoch % params['n_eval_steps'] == 0:
                self._save(experiment)
        return ep_loss / params['n_epochs']


if __name__ == '__main__':
    # Create log dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--env',
        type = str,
        help = 'environment name'
    )
    parser.add_argument(
        '--env_version',
        nargs='?', type = int, const = 1,
        help = 'environment version'
    )
    parser.add_argument(
        '--env_class',
        type = str,
        help = 'entry point to custom environment'
    )
    parser.add_argument(
        '--model_class',
        type = str,
        help = 'entry point to model class'
    )
    parser.add_argument(
        '--render',
        nargs='?', type = int, const = 1,
        help = 'choice to render env'
    )
    args = parser.parse_args()

    path = os.path.join(args.out_path, 'exp{}'.format(args.experiment))
    if not os.path.exists(path):
        os.mkdir(path)
    tensorboard_log = os.path.join(path, 'tensorboard')
    if not os.path.exists(tensorboard_log):
        os.mkdir(tensorboard_log)
    log_dir = path

    env_name = args.env + '-v' + str(args.env_version)
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    env_class = None
    env = None
    import imp
    render = False
    if args.render is not None:
        render = True
    if torch.cuda.is_available():
        render = False
    if env_name in env_ids:
        env = stable_baselines3.common.env_util.make_vec_env(
            env_name,
            monitor_dir = log_dir,
            monitor_kwargs = {
                'info_keywords' : info_kwargs
            },
        )
    if env_name not in env_ids and args.env_class is not None:
        print('Environment Name: {}'.format(env_name))
        gym.envs.registration.register(
            id=env_name,
            entry_point=args.env_class,
            max_episode_steps = params['MAX_STEPS']
        )
        env = gym.make(env_name)

    if not os.path.exists(os.path.join(args.out_path, str(args.experiment), 'tensorboard')):
        os.mkdir(os.path.join(args.out_path, str(args.experiment), 'tensorboard'))
    path = os.path.join(args.out_path, 'exp{}'.format(str(args.experiment)), 'tensorboard')
    if os.path.exists(path):
        remove(path)
        os.mkdir(path)
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join(args.out_path, 'exp{}'.format(str(args.experiment)), 'tensorboard'))
    module_name, class_name = args.model_class.split(':')
    fp, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, fp, pathname, description)
    model_class = module.__dict__[class_name]
    learner = Learner(args.out_path, env, model_class, writer)
    learner.learn(args.experiment)
