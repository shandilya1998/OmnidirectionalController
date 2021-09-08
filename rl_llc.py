import os
import argparse
import gym
import stable_baselines3 as sb
from constants import params
from simulations import QuadrupedV3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class Learner:
    def __init__(self,
        logdir, logger,
        model_class,
        model_kwargs,
        env_class,
        env_kwargs,
        callback_class,
        callback_kwargs,
        modelpath = None,
        envclass = None,
    ):
        self.logdir = logdir
        self.logger = logger
        if modelpath is not None:
            self._model = model_class.load(modelpath)
        else:
            self._model = model_class(**model_kwargs)

        self._env = env_class(**env_kwargs)
        self._callback = callback_class(**callback_kwargs)

    def load_policy(self, module_path):
        self._model.policy.actor.mu = torch.load(module_path)
        self._model.policy.actor_target.mu = torch.load(module_path)


    def learn(self):
        self._model.learn(
            total_timesteps = params['steps'],
            callback = self._callback
        )
        self.model.save(os.path.join(
            self.logdir,
            'Policy'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--logdir',
        type = str,
        help = 'Path to output directory'
    )

    args = parser.parse_args()
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(
        os.path.join(
            args.logdir,
            'exp{}'.format(args.experiment)
        )
    ):
        os.mkdir(
            os.path.join(
                args.logdir,
                'exp{}'.format(str(args.experiment))
            )
        )
    else:
        shutil.rmtree(
            os.path.join(
                args.logdir,
                'exp{}'.format(str(args.experiment))
            )
        )
        os.mkdir(
            os.path.join(
                args.logdir,
                'exp{}'.format(str(args.experiment))
            )
        )
    logdir = os.path.join(args.logdir, 'exp{}'.format(str(args.experiment)))
    if os.path.exists(os.path.join(logdir, 'best_model')):
        shutil.rmtree(os.path.join(logdir, 'best_model'))
        os.mkdir(os.path.join(logdir, 'best_model'))
    else:
        os.mkdir(os.path.join(logdir, 'best_model'))
    logger = tensorboard.SummaryWriter(os.path.join(logdir, 'tensorboard'))

    env_kwargs = {}
    eval_env = QuadrupedV3(**env_kwargs)
    env = QuadrupedV3(**env_kwargs)

    policy = 'MlpPolicy'
    policy_kwargs = {
        'observation_space' : env.observation_space,
        'action_space' : env.action_space
    }

    learner = Learner(
        logdir, logger,
        sb3.TD3,
        {
            'policy' : policy,
            'env' : env,
            'learning_rate' : params['LEARNING_RATE'],
            'buffer_size' : params['buffer_size'],
            'learning_starts' : params['LEARNING_STARTS'],
            'batch_size' : params['BATCH_SIZE'],
            'tau' : params['TAU'],
            'gamma' : params['GAMMA'],
            'train_freq' : params['TRAIN_FREQ'],
        },
        QuadrupedV3,
        {},
        sb3.common.callbacks.EvalCallback,
        {
            'eval_env' : eval_env,
            'n_eval_episodes' : params['n_eval_episodes'],
            'eval_freq' : params['eval_freq'],
            'log_path' : logdir,
            'best_model_save_path' : os.path.join(logdir, 'best_model'),
            'deterministic' : True,
            'render' : False,
            'verbose' : 0
        }
    )
