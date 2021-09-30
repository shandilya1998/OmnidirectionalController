import numpy as np
from constants import params
import gym
from collections import OrderedDict
import copy
import random

def get_control_params(env, run_type = 'random', seed = None):
    prng = np.random.RandomState(seed)
    mu = prng.uniform(
        low = params['props'][env.gait]['mu'][0],
        high = params['props'][env.gait]['mu'][1]
    )
    omega = prng.uniform(
        low = params['props'][env.gait]['omega'][0],
        high = params['props'][env.gait]['omega'][1]
    )
    if run_type == 'constant_mu':
        if env.gait == 'trot':
            mu = 0.45
        else:
            mu = 0.6
    elif run_type == 'constant_omega':
        if env.gait == 'trot':
            omega = 4.4 / (2 * np.pi)
        else:
            omega = 2.2 / (2 * np.pi)
    z = env._get_z()
    omega = np.array(
        [omega] * env._num_legs, dtype = np.float32
    ) * env.heading_ctrl
    if env.task == 'straight' or env.task == 'rotate':
        mu = np.array([mu] * env._num_legs, dtype = np.float32)
    elif env.task == 'turn':
        mu1 = copy.deepcopy(mu)
        mu2 = prng.uniform(
            low = mu,
            high = params['props'][env.gait]['mu'][-1]
        )
        if env.direction == 'left':
            mu = np.array([mu1, mu2, mu2, mu1], dtype = np.float32)
        elif env.direction == 'right':
            mu = np.array([mu2, mu1, mu1, mu2], dtype = np.float32)
    else:
        raise ValueError('Expected one of `straight`, `rotate` or \
            `turn`, got {}'.format(env.task))
    return omega, mu, z

def sample_gait_task_direction(seed = None):
    random.seed(seed)
    direction = None
    task = None
    gait = random.choice(['trot', 'ls_crawl', 'ds_crawl'])
    if 'crawl' in gait:
        task = random.choice(['straight', 'turn', 'rotate'])
    else:
        task = random.choice(['straight', 'turn'])
    if task == 'turn' or task == 'rotate':
        direction = random.choice(['left', 'right'])
    else:
        direction = random.choice([
            'forward', 'backward',
            'left', 'right'
        ])
    return gait, task, direction

def convert_observation_to_space(observation, maximum = float('inf')):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -maximum, dtype=np.float32)
        high = np.full(observation.shape, maximum, dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
