import os
import pandas as pd
from tqdm import tqdm
from constants import params
import shutil

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
    count = 0
    for path, info in zip(datapaths, infos):
        for i in range(len(info)):
            pbar.update(1)
            name = info.iloc[i]['id']
            paths_src = [os.path.join(
                path,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            name = name.split('_')
            name[1] = str(int(name[1]) + count)
            name = '_'.join(name)
            paths_dst = [os.path.join(
                outpath,
                '{}_{}.npy'.format(name, item)
            ) for item in params['track_list']]
            info.at[i, 'id'] = name
            for p_s, p_d in zip(paths_src, paths_dst):
                shutil.copy2(p_s, p_d)
            count += 1
        out_info = pd.concat([out_info, info], ignore_index = True)
    pbar.close()
    out_info.to_csv(os.path.join(outpath, 'info.csv'))
