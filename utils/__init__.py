import torch
import utils.data_generator as data_generator
import utils.networks as networks
import utils.dataset as dataset
#import utils.networks.Controller as Controller
#from utils import data_generator, networks, dataset
from utils.networks import Controller, ControllerV2, \
    ControllerV3, ControllerV4, ControllerV5
from utils.dataset import SupervisedLLCDataset
import numpy as np
import gym
import os
import shutil
from utils.torch_utils import *
from utils.os_utils import remove

def _get_init_gamma(gait, task, direction):
    init_gamma = [0.0, 0.0, 0.0, 0.0]
    heading_ctrl  = np.ones(4, dtype = np.float32)
    if gait == 'ds_crawl':
        init_gamma = [0.0, 0.5, 0.75, 0.25]
    elif gait == 'ls_crawl':
        init_gamma = [0, 0.5, 0.25, 0.75]
    elif gait == 'trot':
        init_gamma = [0, 0.5, 0, 0.5]
    elif gait == 'pace':
        init_gamma = [0, 0.5, 0.5, 0]
    elif gait == 'bound':
        init_gamma = [0, 0, 0.5, 0.5]
    elif gait == 'transverse_gallop':
        init_gamma = [0, 0.1, 0.6, 0.5]
    elif gait == 'rotary_gallop':
        init_gamma = [0, 0.1, 0.5, 0.6]
    if task == 'straight':
        if direction == 'forward':
            heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
        if direction == 'backward':
            init_gamma = init_gamma[-2:] + init_gamma[:2]
            heading_ctrl = np.array([-1.0, 1.0, 1.0, -1.0], dtype = np.float32)
        if direction == 'left':
            init_gamma = init_gamma[1:] + init_gamma[:1]
            heading_ctrl = np.array([1.0, 1.0, -1.0, -1.0], dtype = np.float32)
        if direction == 'right':
            init_gamma = init_gamma[-1:] + init_gamma[:-1]
            heading_ctrl = np.array([-1.0, -1.0, 1.0, 1.0], dtype = np.float32)
    if task == 'rotate':
        if direction == 'left':
            init_gamma = [0.75, 0.5, 0.25, 0.0]
            heading_ctrl = np.array([1.0, 1.0, 1.0, 1.0], dtype = np.float32)
        if direction == 'right':
            init_gamma = [0, 0.25, 0.5, 0.75]
            heading_ctrl = np.array([-1.0, -1.0, -1.0, -1.0], dtype = np.float32)
    if task == 'turn':
        heading_ctrl = np.array([1.0, -1.0, -1.0, 1.0], dtype = np.float32)
    init_gamma = np.array(init_gamma, dtype = np.float32)
    heading_ctrl *= 1.0
    C = np.load('assets/out/plots/coef.npy')
    return init_gamma, heading_ctrl
