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

np.seterr('ignore')
info_kwargs = (
    'reward_velocity',
    'reward_position',
    'reward_energy',
    'reward_ang_vel',
    'reward_orientation',
    'reward_contact',
    'reward_ctrl',
    'reward',
)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, env,batch_size, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.env = env
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    """
    def _on_rollout_start(self):
        self.env.envs[0].reset_state()
    """

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 1:
            df = load_results(self.log_dir)
            if len(df) > 0:
                reward_velocity = np.mean(df.reward_velocity.values[-100:])
                reward_energy = np.mean(df.reward_energy.values[-100:])
                reward = np.mean(df.reward.values[-100:])
                reward_position = np.mean(df.reward_position.values[-100:])
                reward_ang_vel = np.mean(df.reward_ang_vel.values[-100:])
                reward_orientation = np.mean(df.reward_ang_vel.values[-100:])
                reward_contact = np.mean(df.reward_contact.values[-100:])
                reward_ctrl = np.mean(df.reward_ctrl.values[-100:])
                self.logger.record('reward_velocity', reward_velocity)
                self.logger.record('reward_energy', reward_energy)
                self.logger.record('reward', reward)
                self.logger.record('reward_ang_vel', reward_ang_vel)
                self.logger.record('reward_position', reward_position)
                self.logger.record('reward_orientation', reward_orientation)
                self.logger.record('reward_contact', reward_contact)
                self.logger.record('reward_ctrl', reward_ctrl)
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            df = load_results(self.log_dir)
            x, y = ts2xy(df, 'timesteps')
            if len(x) > 0:
            # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    #print(stable_baselines3.common.evaluation.evaluate_policy(self.model, self.env, render=True))
        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    fig.savefig(os.path.join(log_folder, 'learning_curve.png'))


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
    if env_name in env_ids:
        env = stable_baselines3.common.env_util.make_vec_env(
            env_name,
            monitor_dir = log_dir,
            monitor_kwargs = {
                'info_keywords' : info_kwargs
            },
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
            n_envs = params['n_envs'],
            monitor_dir = log_dir,
            monitor_kwargs = {
                'info_keywords' : info_kwargs
            },
            env_kwargs = {
                'model_path' : 'ant.xml',
                'render' : render,
                'gait' : 'trot',
                'task' : 'straight',
                'direction' : 'left',
                'policy_type' : 'MultiInputPolicy',
                'track_lst' : params['track_list'],
                'verbose' : 1
            }
        )

    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.sample().shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean = params['OU_MEAN'] * np.ones(n_actions), sigma= params['OU_SIGMA']* np.ones(n_actions))
    # Create the callback: check every 1000 steps
    # Create RL model

    #batch_size = 128

    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(env, params['BATCH_SIZE'], check_freq = params['CHECK_FREQ'], log_dir=log_dir, verbose = 1)

    policy_kwargs = {}
    policy_kwargs['net_arch'] = params['NET_ARCH']
    policy_kwargs['activation_fn'] = torch.nn.Tanh
    policy = params['POLICY_TYPE']
    if args.ppo is not None:
        policy_kwargs['log_std_init'] = -1
        policy_kwargs['ortho_init'] = False

    model = None
    if args.td3 is not None:
        if args.her is None:
            print(params['TRAIN_FREQ'])
            model = TD3(
                policy,
                env,
                policy_kwargs = policy_kwargs,
                learning_starts = params['LEARNING_STARTS'],
                action_noise = action_noise,
                verbose = 1,
                tensorboard_log = log_dir,
                batch_size = params['BATCH_SIZE'],
                train_freq = params['TRAIN_FREQ'][0],
            )
        else:
            print('[DDPG] Using HER')
            model = TD3(
                policy,
                env,
                policy_kwargs = policy_kwargs,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs = dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length = params['MAX_STEPS'],
                ),
                learning_starts = params['LEARNING_STARTS'],
                action_noise = action_noise,
                verbose = 1,
                tensorboard_log = log_dir,
                batch_size = params['BATCH_SIZE'],
                train_freq = params['TRAIN_FREQ'][0],
            )

    elif args.ppo is not None:
        model = PPO(
            policy,
            env,
            policy_kwargs = policy_kwargs,
            tensorboard_log = log_dir,
            verbose = 1,
            batch_size = params['BATCH_SIZE'],
            use_sde = True,
            sde_sample_freq = params['sde_sample_freq'], # 4
            n_epochs = params['n_epochs'], # 20
            n_steps = params['MAX_STEPS'],
            gae_lambda = params['gae_lambda'],#0.9,
            clip_range = params['clip_range'],#0.4,
            device='cuda'
        )
    elif args.a2c is not None:
        model = A2C(
            policy,
            env = env,
            policy_kwargs = policy_kwargs,
            tensorboard_log = log_dir,
            verbose = 1,
            n_steps = params['n_steps'], #8,
            gae_lambda = params['gae_lambda'], # 0.9
            vf_coef = params['vf_coef'], #0.4,
            learning_rate = params['LEARNING_RATE'],
            use_sde = True,
            sde_sample_freq = params['sde_sample_freq'],
            normalize_advantage=True,
        )
    elif args.sac is not None:
        if args.her is None:
            model = SAC(
                policy,
                env,
                learning_starts=params['LEARNING_STARTS'],
                action_noise = action_noise,
                verbose = 1,
                tensorboard_log = log_dir,
                batch_size = params['BATCH_SIZE'],
                gamma = params['gamma'],
                tau = params['tau'],
                train_freq = params['TRAIN_FREQ'],
                use_sde = True
            )
        else:
            print('[DDPG] Using HER')
            model = SAC(
                policy,
                env,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs = dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length = params['MAX_STEPS'],
                ),
                learning_starts=params['LEARNING_STARTS'],
                action_noise = action_noise,
                verbose = 1,
                tensorboard_log = log_dir,
                batch_size = params['BATCH_SIZE'],
                gamma = params['gamma'], #0.98,
                tau = params['tau'], #0.02,
                train_freq = params['TRAIN_FREQ'],
                use_sde = True,
            )


    steps = params['steps']
    model.learn(total_timesteps=int(steps), callback=callback)
    model.save(log_dir + '/Policy')
    if args.ppo is not None or args.a2c is not None:
        torch.save(model.policy.action_net, os.path.join(log_dir, 'action_net.pth'))
        torch.save(model.policy.value_net, os.path.join(log_dir, 'value_net.pth'))
    elif args.sac is not None:
        torch.save(model.actor, os.path.join(log_dir, 'actor.pth'))
        torch.save(model.critic, os.path.join(log_dir, 'critic.pth'))
        torch.save(model.critic_target, os.path.join(log_dir, 'critic_target.pth'))
    else:
        torch.save(model.actor, os.path.join(log_dir, 'actor.pth'))
        torch.save(model.critic, os.path.join(log_dir, 'critic.pth'))
        torch.save(model.critic_target, os.path.join(log_dir, 'critic_target.pth'))
        torch.save(model.actor_target, os.path.joint(log_dir, 'actor_target.pth'))


    from stable_baselines3.common import results_plotter

    # Helper from the library
    #results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DDPG Ant-v4")
    print(stable_baselines3.common.evaluation.evaluate_policy(model, env, render=True))
    print('DONE')
