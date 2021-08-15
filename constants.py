import torch
import numpy as np
import os


action_dim = 8
units_osc = action_dim #60#action_dim#60 exp 68 units_osc = 8
cpg_param_size = 16
input_size = 6
params = {
    'motion_state_size'           : 6,#:exp69, 6,#:exp 67,68, 3 #:exp66, 4 :exp64,65,
    'robot_state_size'            : 45,#:exp69, 111,#:exp67,68, 111 for stable_baselines model #4*action_dim + 4 + 8*3,
    'dt'                          : 0.001,
    'units_output_mlp'            : [60,100, 100, cpg_param_size],
    'input_size_low_level_control': input_size, 
    'units_low_level_control'     : [30, 300, 500, 250, 40],
    'cpg_param_size'              : cpg_param_size,
    'units_osc'                   : units_osc,
    'units_combine_rddpg'         : [200, units_osc],
    'units_combine'               : [200, units_osc],
    'units_robot_state'           : [145, 250, units_osc],
    'units_motion_state'          : [150, 75],
    'units_mu'                    : [150, 75],
    'units_mean'                  : [150, 75],
    'units_omega'                 : [150, 75],
    'units_dim_omega'             : 1,
    'units_robot_state_critic'    : [200, 120],
    'units_gru_rddpg'             : 400,
    'units_q'                     : 1,
    'actor_version'               : 2,
    'hopf_version'                : 1,
    'units_motion_state_critic'   : [200, 120],
    'units_action_critic'         : [200, 120],
    'units_history'               : 24,
    'BATCH_SIZE'                  : 200,
    'BUFFER_SIZE'                 : 1000000,
    'WARMUP'                      : 20000,
    'GAMMA'                       : 0.99,
    'TEST_AFTER_N_EPISODES'       : 25,
    'TAU'                         : 0.001,
    'decay_steps'                 : int(20),
    'LRA'                         : 1e-3,
    'LRC'                         : 1e-3,
    'EXPLORE'                     : 10000,
    'train_episode_count'         : 20000000,
    'test_episode_count'          : 5,
    'max_steps'                   : 1,
    'action_dim'                  : action_dim,
    'action_scale_factor'         : 1.2217,#:exp69, 1.0,
    'scale_factor_1'              : 0.1,
    'units_action_input'          : 20,
    'rnn_steps'                   : 128,
    'units_critic_hidden'         : 20,
    'lstm_units'                  : action_dim,
    'lstm_state_dense_activation' : 'relu',
    'L0'                          : 0.01738,
    'L1'                          : 0.025677,
    'L2'                          : 0.017849,
    'L3'                          : 0.02550,
    'g'                           : -9.81,
    'thigh'                       : 0.06200,
    'base_breadth'                : 0.04540,
    'friction_constant'           : 1e-4,
    'mu'                          : 0.001,
    'm1'                          : 0.010059,
    'm2'                          : 0.026074,
    'm3'                          : 0.007661,
    'future_steps'                : 4,
    'ou_theta'                    : 0.15,
    'ou_sigma'                    : 0.2,
    'ou_mu'                       : 0.0,
    'seed'                        : 1,
    'trajectory_length'           : 20,
    'window_length'               : 6,
    'num_validation_episodes'     : 5,
    'validate_interval'           : 20000,
    'update_interval'             : 2,
    'step_version'                : 1,
    'transition_items'            : ['ob','action','reward','next_ob','done', 'info'],
    'her_ratio'                   : 0.4,
    'render_train'                : True,
    'log_interval'                : 5000,
}

class TensorSpec:
    def __init__(self, shape, dtype, name = None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

observation_spec = [
    TensorSpec(
        shape = (
            params['motion_state_size'],
        ),
        dtype = np.float32,
        name = 'motion_state_inp'
    ),
    TensorSpec(
        shape = (
            params['robot_state_size'],
        ),
        dtype = np.float32,
        name = 'robot_state_inp'
    ),
    TensorSpec(
        shape = (
            params['units_osc'] * 2,
        ),
        dtype = np.float32,
        name = 'oscillator_state_inp'
    ),
]

params_spec = [
    TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = np.float32,
        name = 'quadruped action'
    )
] * 2

action_spec = [
    TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = np.float32,
        name = 'quadruped action'
    ),
   TensorSpec(
       shape = (
           params['rnn_steps'],
           params['units_osc'] * 2,
       ),
       dtype = np.float32,
       name = 'oscillator action'
   )
]

reward_spec = [
    TensorSpec(
        shape = (),
        dtype = np.float32,
        name = 'reward'
    )
]

history_spec = TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        params['action_dim']
    ),
    dtype = np.float32
)

history_osc_spec = TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        2* params['units_osc']
    ),
    dtype = np.float32
)


data_spec = []
data_spec.extend(observation_spec)
data_spec.extend(action_spec)
data_spec.extend(params_spec)
data_spec.extend(reward_spec)
data_spec.append(observation_spec)

data_spec.extend([
    TensorSpec(
        shape = (),
        dtype = np.bool,
        name = 'done'
    )
])

specs = {
    'observation_spec' : observation_spec,
    'reward_spec' : reward_spec,
    'action_spec' : action_spec,
    'data_spec' : data_spec,
    'history_spec' : history_spec,
    'history_osc_spec' : history_osc_spec
}

params.update(specs)

robot_data = {
    'leg_name_lst' : [
        'front_right_leg',
        'front_left_leg',
        'back_right_leg',
        'back_left_leg'
    ],
    'link_name_lst' :  [
        'quadruped::base_link',
        'quadruped::front_right_leg1',
        'quadruped::front_right_leg2',
        'quadruped::front_right_leg3',
        'quadruped::front_left_leg1',
        'quadruped::front_left_leg2',
        'quadruped::front_left_leg3',
        'quadruped::back_right_leg1',
        'quadruped::back_right_leg2',
        'quadruped::back_right_leg3',
        'quadruped::back_left_leg1',
        'quadruped::back_left_leg2',
        'quadruped::back_left_leg3'
    ],
    'joint_name_lst' : [
        'front_right_leg1_joint',
        'front_right_leg2_joint',
        'front_right_leg3_joint',
        'front_left_leg1_joint',
        'front_left_leg2_joint',
        'front_left_leg3_joint',
        'back_right_leg1_joint',
        'back_right_leg2_joint',
        'back_right_leg3_joint',
        'back_left_leg1_joint',
        'back_left_leg2_joint',
        'back_left_leg3_joint'
    ],
    'starting_pos' : np.array([
        -0.01, 0.01, 0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, 0.01
    ], dtype = np.float32),
    'L' : 2.2*0.108,
    'W' : 2.2*0.108
}

params.update(robot_data)

Tsw = [i for i in range(80, 260, 4)]
Tsw = Tsw + Tsw + Tsw
Tst = [i*3 for i in Tsw]
Tst = Tst + Tst + Tst
theta_h = [30 for i in range(len(Tsw))] + [45 for i in range(len(Tsw))] + \
        [60 for i in range(len(Tsw))]
theta_k = [30 for i in range(len(Tsw))] + [30 for i in range(len(Tsw))] + \
        [30 for i in range(len(Tsw))]

pretraining = {
    'Tst' : Tst,
    'Tsw' : Tsw,
    'theta_h' : theta_h,
    'theta_k' : theta_k
}


num_data = 135
params.update(pretraining)
bs = 15
#num_data =135 * params['rnn_steps'] * params['max_steps']
params.update({
    'num_data' : num_data,
    'pretrain_bs': bs,
    'train_test_split' : (num_data - bs) / num_data,
    'pretrain_test_interval' : 3,
    'pretrain_epochs' : 500,
    'pretrain_ds_path': 'data/pretrain_rddpg_6'
    })


params_ars = {
    'nb_steps'                    : 1000,
    'episode_length'              : 300,
    'learning_rate'               : 0.001,
    'nb_directions'               : 12,
    'rnn_steps'                   : 1,
    'nb_best_directions'          : 7,
    'noise'                       : 0.0003,
    'seed'                        : 1,
}

params_per = {
    'alpha'                       : 0.6,
    'epsilon'                     : 1.0,
    'beta_init'                   : 0.4,
    'beta_final'                  : 1.0,
    'beta_final_at'               : params['train_episode_count'],
    'step_size'                   : params['LRA'] / 4
}

params.update(params_per)

params_env = {
    'DEFAULT_SIZE'                : 500,
    'INIT_HEIGHT'                 : 0.05,
    'dt'                          : 0.001,
    'reward_energy_coef'          : 1e-2,
    'reward_velocity_coef'        : 1e2,
    'INIT_JOINT_POS'              : np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], dtype = np.float32)
}


max_steps = 300 # 150 #data collection env
n_envs = 1
params_rl = {
    'MAX_STEPS'                   : max_steps,
    'OU_MEAN'                     : 0.0,
    'OU_SIGMA'                    : 0.08,
    'BATCH_SIZE'                  : 10, #max_steps * n_envs ,#8,
    'NET_ARCH'                    : [dict(pi=[256, 256], vf=[256, 256])],
    'POLICY_TYPE'                 : "MultiInputPolicy",
    'LEARNING_STARTS'             : 5000,
    'TRAIN_FREQ'                  : [1, 'steps'],
    'CHECK_FREQ'                  : 6000,
    'sde_sample_freq'             : 4,
    'n_epochs'                    : 15,
    'gae_lambda'                  : 0.9,
    'clip_range'                  : 0.4,
    'vf_coef'                     : 0.4,
    'LEARNING_RATE'               : 0.001,
    'gamma'                       : 0.9,
    'tau'                         : 0.02,
    'steps'                       : int(1e7),
    'n_steps'                     : 8,
    'n_envs'                      : n_envs,
    'dt'                          : 0.001,
    'max_step_length'             : 600,
    'gait_list'                   : [
                                        'ds_crawl',
                                        'ls_crawl',
                                        'trot',
                                        'pace',
                                        'bound',
                                        'transverse_gallop',
                                        'rotary_gallop'
                                    ],
    'task_list'                   : [
                                        'straight',
                                        'rotate',
                                        'turn',
                                    ],
    'direction_list'              : [
                                        'forward',
                                        'backward',
                                        'left',
                                        'right'
                                    ],
    'track_list'                  : [
                                        'joint_pos',
                                        'action',
                                        'velocity',
                                        'position',
                                        'true_joint_pos',
                                        'sensordata',
                                        'qpos',
                                        'qvel',
                                        'achieved_goal',
                                        'observation',
                                        'desired_goal',
                                        'heading_ctrl',
                                        'omega',
                                        'z',
                                        'mu',
                                        'd1',
                                        'd2',
                                        'd3',
                                        'stability',
                                        'omega_o'
                                    ],
    'ref_path'                    : os.path.join('assets', 'out', 'reference'),
    'env_name'                    : 'Quadruped',
    'osc_omega_offset'            : 0.0
}


action_dim = 8
units_osc = 60#action_dim#60 exp 68 units_osc = 8
params_pretrain = {
    'action_dim'                  : action_dim,
    'batch_size'                  : 512,
    'n_epochs'                    : 15000,
    'n_steps'                     : 3500,
    'n_update_steps'              : 20,
    'n_eval_steps'                : 100,
    'n_episodes'                  : 500,
    'min_epoch_size'              : 200,
    'motion_state_size'           : 6,#:exp69, 6,#:exp 67,68, 3 #:exp66, 4 :exp64,65,
    'robot_state_size'            : 18,#:exp69, 111,#:exp67,68, 111 for stable_baselines model #4*action_dim + 4 + 8*3,
    'dt'                          : 0.001,
    'units_output_mlp'            : [256,512,100,action_dim],
    'units_osc'                   : units_osc,
    'units_combine_rddpg'         : [200, units_osc],
    'units_combine'               : [200, units_osc],
    'units_robot_state'           : [145, 250, units_osc],
    'units_motion_state'          : [150, 75],
    'units_mu'                    : [150, 75],
    'units_mean'                  : [150, 75],
    'units_omega'                 : [150, 75],
    'units_dim_omega'             : 1,
    'units_robot_state_critic'    : [200, 120],
    'units_gru_rddpg'             : 400,
    'units_q'                     : 1,
    'actor_version'               : 2,
    'hopf_version'                : 1,
    'units_motion_state_critic'   : [200, 120],
    'units_action_critic'         : [200, 120],
    'units_history'               : 24,
    'max_action'                  : 2.0,
}

params_conv = {
    'h'                           : 250,
    'w'                           : params_pretrain['motion_state_size'],
    'stride'                      : 100,
    'window_length'               : 2500,
}

params.update(params_env)
params.update(params_rl)
params.update(params_pretrain)
params.update(params_conv)

params.update({
    'version'                     : 1,
    'offset'                      : np.array([0.25, 0.25, 0.25, 0.25], dtype = np.float32),
    'degree'                      : 15,
    'thresholds'                  : np.array([
                                        0.0,
                                        np.pi / 6,
                                        5 * np.pi / 6,
                                        np.pi,
                                        11 * np.pi / 6 - np.pi / 10,
                                        11 * np.pi / 6,
                                        2 * np.pi
                                    ], dtype = np.float32),
    'camera_name'                 : 'esp32cam',
})

params['gait_list'] = ['trot', 'ds_crawl', 'ls_crawl']
params['task_list'] = ['straight', 'rotate', 'turn']
params['direction_list'] = ['forward', 'backward', 'left', 'right']
