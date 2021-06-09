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
        '--her',
        nargs='?', type = int, const = 1,
        help = 'choice to use HER replay buffer'
    )
    parser.add_argument(
        '--td3',
        nargs='?', type = int, const = 1,
        help = 'choice to use TD3'
    )
    parser.add_argument(
        '--ppo',
        nargs='?', type = int, const = 1,
        help = 'choice to use PPO'
    )
    parser.add_argument(
        '--sac',
        nargs='?', type = int, const = 1,
        help = 'choice to use SAC'
    )
    parser.add_argument(
        '--a2c',
        nargs='?', type = int, const = 1,
        help = 'choice to use A2C'
    )
    parser.add_argument(
        '--model_dir',
        type = str,
        help = 'path to model files'
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
    if env_name in env_ids:
        env = stable_baselines3.common.env_util.make_vec_env(
            env_name,
            monitor_dir = log_dir,
        )
    if env_name not in env_ids and args.env_class is not None:
        gym.envs.registration.register(
            id=env_name,
            entry_point=args.env_class,
            max_episode_steps = params['MAX_STEPS']
        )
        """
        module_name, class_name = args.env_class.split(':')
        fp, pathname, description = imp.find_module(module_name)
        module = imp.load_module(module_name, fp, pathname, description)
        env_class = module.__dict__[class_name]
        """
        env = stable_baselines3.common.env_util.make_vec_env(
            env_name,
            n_envs = 1,
            monitor_dir = log_dir,
            env_kwargs = {
                'model_path' : 'ant.xml',
                'render' : False,
                'gait' : 'trot',
                'task' : 'straight',
                'direction' : 'forward',
                'policy_type' : 'MultiInputPolicy',
                'track_lst' : ['joint_pos', 'action', 'velocity', 'position', 'true_joint_pos']
            }
        )


    if args.td3 is not None:
        model = TD3.load(os.path.join(args.model_dir, 'Policy'))
        actor = torch.load(os.path.join(args.model_dir, 'actor.pth'), map_location=torch.device('cpu'))
        critic = torch.load(os.path.join(args.model_dir, 'critic.pth'), map_location=torch.device('cpu'))
        actor_target = torch.load(os.path.join(args.model_dir, 'actor_target.pth'), map_location=torch.device('cpu'))
        critic_target = torch.load(os.path.join(args.model_dir, 'critic_target.pth'), map_location=torch.device('cpu'))
        model.policy.actor.load_state_dict(actor.state_dict(), strict = False)
        model.policy.actor_target.load_state_dict(actor_target.state_dict(), strict = False)
        model.policy.critic.load_state_dict(critic.state_dict(), strict = False)
        model.policy.critic_target.load_state_dict(critic_target.state_dict(), strict = False)

    elif args.sac is not None:
        model = SAC.load(os.path.join(args.model_dir, 'Policy'))
        actor = torch.load(os.path.join(args.model_dir, 'actor.pth'), map_location=torch.device('cpu'))
        critic = torch.load(os.path.join(args.model_dir, 'critic.pth'), map_location=torch.device('cpu'))
        critic_target = torch.load(os.path.join(args.model_dir, 'critic_target.pth'), map_location=torch.device('cpu'))
        actor_state_dict = actor.state_dict()
        critic_state_dict = critic.state_dict()
        critic_target_state_dict = critic_target.state_dict()
        model.policy.actor.load_state_dict(actor_state_dict, strict = False)
        model.policy.critic.load_state_dict(critic_state_dict, strict = False)
        model.policy.critic_target.load_state_dict(critic_target_state_dict, strict = False)

    elif args.a2c is not None:
        model = A2C.load(os.path.join(args.model_dir, 'Policy'))
        action_net = torch.load(os.path.join(args.model_dir, 'action_net.pth'), map_location=torch.device('cpu'))
        value_net = torch.load(os.path.join(args.model_dir, 'value_net.pth'), map_location=torch.device('cpu'))
        model.policy.action_net.load_state_dict(action_net.state_dict(), strict = False)
        model.policy.value_net.load_state_dict(value_net.state_dict(), strict = False)
    elif args.ppo is not None:
        model = PPO.load(os.path.join(args.model_dir, 'Policy'))
        action_net = torch.load(os.path.join(args.model_dir, 'action_net.pth'), map_location=torch.device('cpu'))
        value_net = torch.load(os.path.join(args.model_dir, 'value_net.pth'), map_location=torch.device('cpu'))
        model.policy.action_net.load_state_dict(action_net.state_dict(), strict = False)
        model.policy.value_net.load_state_dict(value_net.state_dict(), strict = False)

    print(stable_baselines3.common.evaluation.evaluate_policy(model, env, render=True))
    print('DONE')
