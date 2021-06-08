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

info_kwargs = (
    'reward_velocity',
    'reward_energy',
    'reward',
    'penalty'
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
                penalty = np.mean(df.penalty.values[-100:])
                self.logger.record('reward_velocity', reward_velocity)
                self.logger.record('reward_energy', reward_energy)
                self.logger.record('reward', reward)
                self.logger.record('penalty', penalty)
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
    if env_name not in env_ids and args.env_class is not None:
        gym.envs.registration.register(
            id=env_name,
            entry_point=args.env_class,
            max_episode_steps = params['MAX_STEPS']
        )
    elif env_name not in env_ids:
        raise ValueError('provide an entrypoint')
    else:
        raise ValueError('invalid env name')


    env = stable_baselines3.common.env_util.make_vec_env(
        env_name,
        monitor_dir = log_dir,
        monitor_kwargs = {
            'info_keywords' : info_kwargs
        },
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
    policy_kwargs['activation_fn'] = torch.nn.PReLU
    policy = params['POLICY_TYPE']

    model = None
    if args.td3 is not None:
        params['TRAIN_FREQ'] = params['TRAIN_FREQ'][0]
        if args.her is None:
            model = TD3(
                policy,
                env,
                policy_kwargs = policy_kwargs,
                learning_starts = params['LEARNING_STARTS'],
                action_noise = action_noise,
                verbose = 1,
                tensorboard_log = log_dir,
                batch_size = params['BATCH_SIZE'],
                train_freq = params['TRAIN_FREQ'],
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
                train_freq = params['TRAIN_FREQ'],
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
            clip_range = params['clip_range']#0.4,
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
        params['TRAIN_FREQ'] = params['TRAIN_FREQ'][0]
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
