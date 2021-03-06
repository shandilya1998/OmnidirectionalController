from utils.data_generator import generate_multi_goal_gait_data_v2, \
    generate_multi_goal_gait_data, generate_multi_goal_gait_data_v3
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
    parser.add_argument(
        '--render',
        nargs='?', type = int, const = 1,
        help = 'choice to render env'
    )
    parser.add_argument(
        '--version',
        type = int,
        help = 'data generator version'
    )
    args = parser.parse_args()

    print('Module Initialized.')

    module_name, class_name = args.env_class.split(':')
    fp, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, fp, pathname, description)
    env_class = module.__dict__[class_name]

    render = False
    if args.render is not None:
        render = True

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    env_kwargs = {
        'model_path' : 'ant.xml',
        'render' : render,
        'verbose' : 0
    }
    print('Starting.')
    if args.version == 0:
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
    elif args.version == 1:
        generate_multi_goal_gait_data_v2(
            args.log_dir,
            env_class,
            env_kwargs,
            params['gait_list'],
            params['task_list'],
            params['direction_list'],
            params['track_list'],
            class_name
        )
    elif args.version == 2:
        generate_multi_goal_gait_data_v3(
            args.log_dir,
            env_class,
            env_kwargs,
            params['gait_list'],
            params['task_list'],
            params['direction_list'],
            params['track_list'],
            class_name
        )
    print('Thank you.')
