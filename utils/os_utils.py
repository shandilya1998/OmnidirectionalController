import os
import pandas as pd
from tqdm import tqdm
from constants import params

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

