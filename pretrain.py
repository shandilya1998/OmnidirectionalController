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

np.seterr('ignore')

import sys

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
    def __init__(self, out_path, env, model_class):
        self.log_dir = out_path
        self.env = env
        self._epoch = 0
        self._step = 0
        self._model = model_class()
        self._optim  = torch.optim.Adam(
            self._model.parameters(),
            lr = params['LEARNING_RATE']
        )
        self._mse = torch.nn.MSELoss()

    def _get_data(self):
        track_list = params['track_list']
        commands = self.env.ref_info.sample(params['BATCH_SIZE'])
        files = commands['id'].values.tolist()
        lengths = commands['length'].values.tolist()
        steps = min(lengths)
        desired_goal = np.stack([np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, 'desired_goal')))[:steps, :] for f in files], axis = 0)
        achieved_goal = np.stack([np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, 'achieved_goal')))[:steps, :] for f in files], axis = 0)
        ob = np.stack([np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, 'observation')))[:steps, :] for f in files], axis = 0)
        x = [desired_goal, achieved_goal, ob]
        x = [ to_tensor(item) for item in x]
        y = np.stack([np.load(os.path.join(params['ref_path'], '{}_{}.npy'.format(f, 'joint_pos')))[:steps, :] for f in files], axis = 0)
        osc = np.concatenate([np.ones((params['BATCH_SIZE'], 2 * params['units_osc']), dtype = np.float32)], -1)
        return x, to_tensor(y), to_tensor(osc), steps

    def _save(self, experiment):
        torch.save(self._model, os.path.join(self.log_dir, 'exp{}'.format(experiment),'controller.pth'))

    def _eval(self):
        ob = self.env.reset()
        osc = to_tensor(np.concatenate([np.ones((1, params['units_osc']), dtype = np.float32), np.zeros((1, params['units_osc']), dtype = np.float32)], axis = -1))
        total_reward = 0.0
        done = False
        while not done:
            out = self._model([
                to_tensor(np.expand_dims(ob['desired_goal'], 0)),
                to_tensor(np.expand_dims(ob['achieved_goal'], 0)),
                to_tensor(np.expand_dims(ob['observation'], 0))
            ], osc)
            ac, osc = torch.split(out, [params['action_dim'], 2 * params['units_osc']], -1)
            ob, reward, done, info = env.step(ac[0].detach().cpu().numpy())
            total_reward += reward
        return total_reward

    def learn(self, experiment, logger):
        print('Start Pretraining.')
        ep = 0
        while ep < params['n_episodes']:
            loss = self.pretrain(experiment, logger, ep)
            print('Episode {} Loss {:.6f}'.format(self._epoch, loss))
            logger.add_scalar('Train/Episode Loss', loss, ep)
        print('Pretraining Done.')

    def pretrain(self, experiment, logger, ep):
        """
            modify according to need
        """
        steps = 3500
        ep_loss = 0.0
        while self._epoch < params['n_epochs']:
            x, y, osc, steps = self._get_data()
            epoch_loss = 0.0
            self._step = 0
            while self._step < steps - 1:
                it = 0
                loss = 0.0
                self._model.zero_grad()
                while it < params['n_update_steps'] and self._step < steps - 1:
                    out = self._model([item[:, self._step, :] for item in x], osc)
                    y_pred, osc = torch.split(out, [params['action_dim'], 2 * params['units_osc']], -1)
                    loss += self._mse(y_pred, y[:, self._step, :])
                    it += 1
                    self._step += 1
                loss.backward()
                self._optim.step()
                self._optim.zero_grad()
                osc = osc.detach()
                logger.add_scalar('Train/Loss', loss.detach().cpu().numpy(), (ep + 1) * (self._epoch + 1 ) * self._step)
                epoch_loss += loss.detach().cpu().numpy()
            self._epoch += 1
            if steps > 0:
                epoch_loss = epoch_loss / steps
            ep_loss += epoch_loss
            logger.add_scalar('Train/Epoch Loss', epoch_loss, (ep + 1) * self._epoch)
            if (ep + 1 ) * self._epoch % params['n_eval_steps'] == 0:
                logger.add_scalar('Evaluate/Reward', self._eval(), int(self._epoch / params['n_eval_steps']))
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
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join(args.out_path, str(args.experiment), 'tensorboard'))
    module_name, class_name = args.model_class.split(':')
    fp, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, fp, pathname, description)
    model_class = module.__dict__[class_name]
    learner = Learner(args.out_path, env, model_class)
    learner.learn(args.experiment, writer)
