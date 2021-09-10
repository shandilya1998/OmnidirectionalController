import os
import pandas as pd
from tqdm import tqdm
from constants import params
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def _concat_results(datapath1, datapath2):
    info1 = pd.read_csv(os.path.join(
        datapath1,
        'info.csv'
    ), index_col = 0)

    info2 = pd.read_csv(os.path.join(
        datapath2,
        'info.csv'
    ), index_col = 0)

    last = len(info1)

    for i in tqdm(range(len(info2))):
        name = info2.iloc[i]['id']
        path2 = [os.path.join(
            datapath2,
            '{}_{}.npy'.format(name, item)
        ) for item in params['track_list']]
        name = name.split('_')
        name[1] = str(int(name[1]) + last)
        name = '_'.join(name)
        path1 = [os.path.join(
            datapath1,
            '{}_{}.npy'.format(name, item)
        ) for item in params['track_list']]
        info2.at[i, 'id'] = name
        for p2, p1 in zip(path2, path1):
            os.rename(p2, p1)
    df = pd.concat([info1, info2], ignore_index = True)
    df.to_csv(os.path.join(datapath2, 'info_.csv'))


def _concat_results_v2(datapaths, outpath):
    num_dir = len(datapaths)
    infos = [pd.read_csv(os.path.join(
        path,
        'info.csv'
    ), index_col = 0) for path in datapaths]

    total = 0
    for info in infos:
        total += len(info)

    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    pbar = tqdm(total = total)
    out_info = pd.DataFrame([], columns = infos[0].columns)
    for path, info in zip(datapaths, infos):
        last = len(out_info)
        for i in range(len(info)):
            pbar.update(1)
            name = info.iloc[i]['id']
            paths_src = [os.path.join(
                path,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            name = name.split('_')
            name[1] = str(int(name[1]) + last)
            name = '_'.join(name)
            paths_dst = [os.path.join(
                outpath,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            info.at[i, 'id'] = name
            for p_s, p_d in zip(paths_src, paths_dst):
                shutil.copy2(p_s, p_d)
        out_info = pd.concat([out_info, info], ignore_index = True)
    pbar.close()
    out_info.to_csv(os.path.join(outpath, 'info.csv'))


def _condition(gait, task, direction):
    if task == 'straight' and direction in ['left', 'right']:
        return False
    else:
        return True

def _filter_results(condition, logdir, datapath):
    """
        condition: method with the following arguments:
            - gait: str
            - task: str
            - direction: str
    """
    info = pd.read_csv(os.path.join(datapath, 'info.csv'), index_col = 0)
    df = pd.DataFrame([], columns = info.columns)
    count = 0
    pbar = tqdm(total=len(info))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.mkdir(logdir)
    for index, row in info.iterrows():
        pbar.update(1)
        if not condition(row['gait'], row['task'], row['direction']):
            continue
        else:
            name = row['id']
            paths_src = [os.path.join(
                datapath,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            name = name.split('_')
            name[1] = str(count)
            name = '_'.join(name)
            paths_dst = [os.path.join(
                logdir,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            count += 1
            row['id'] = name
            df = df.append(row, ignore_index = True)
            for p_s, p_d in zip(paths_src, paths_dst):
                shutil.copy2(p_s, p_d)
    pbar.close()
    df.to_csv(os.path.join(logdir, 'info.csv'))
