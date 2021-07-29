from simulations import Quadruped
import numpy as np
import pandas as pd

class QuadrupedV3(gym.GoalEnv, utils.EzPickle):
    def __init__(self,
                 model_path = 'ant.xml',
                 frame_skip = 5,
                 render = False,
                 gait = 'trot',
                 task = 'straight',
                 direction = 'forward',
                 policy_type = 'MultiInputPolicy',
                 track_lst = [
                     'desired_goal', 'joint_pos', 'action',
                     'velocity', 'position', 'true_joint_pos',
                     'sensordata', 'qpos', 'qvel',
                     'achieved_goal', 'observation', 'heading_ctrl',
                     'omega', 'z', 'mu',
                     'd1', 'd2', 'd3',
                     'stability',
                 ],
                 stairs = False,
                 verbose = 0):
        super(QuadrupedV3, self).__init__(
            model_path,
            frame_skip,
            render,
            gait,
            task,
            direction,
            policy_type,
            track_lst,
            stairs,
            verbose
        )
