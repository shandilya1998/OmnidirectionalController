from utils.data_generator import generate_multi_goal_gait_data
import argparse
import imp
from constants import params
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type = str,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--env_class',
        type = str,
        help = 'Entrypoint of environment class'
    )
    args = parser.parse_args()

    module_name, class_name = args.env_class.split(':')
    fp, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, fp, pathname, description)
    env_class = module.__dict__[class_name]

    env_kwargs = {
        'model_path' : 'ant.xml',
        'render' : False,
        'verbose' : 0
    }
    generate_multi_goal_gait_data(
        args.log_dir,
        env_class,
        env_kwargs,
        params['gait_list'],
        params['task_list'],
        params['direction_list'],
        params['track_list'],
        class_name
    )
