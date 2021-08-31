import torch
import numpy as np
import os
import h5py

class SupervisedLLCDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        data = h5py.File(os.path.join(datapath, 'data.hdf5'), "r")
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
