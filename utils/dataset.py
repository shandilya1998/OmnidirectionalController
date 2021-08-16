import torch
import numpy as np
import os

class SupervisedLLCDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        X = torch.from_numpy(np.load(os.path.join(datapath, 'X.npy')))
        Y = torch.from_numpy(np.load(os.path.join(datapath, 'Y.npy')))
        self.data = list(zip(X, Y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
