import torch
import os
from constants import params
import shutil
import argparse
from torch.utils import tensorboard
from tqdm import tqdm
from simulations import QuadrupedV3, QuadrupedV4
import numpy as np
import h5py
from utils.networks import HopfEnsemble

device = torch.device('cuda') \
    if torch.cuda.is_available() \
    else torch.device('cpu')

class Learner:
    def __init__(self, logdir, datapath, logger):
        self.logdir = logdir
        self.datapath = datapath
        self.logger = logger
        self._model = Controller()
        self._optim  = torch.optim.Adam(
            self._model.parameters(),
            lr = params['LEARNING_RATE']
        )
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optim,
            gamma = params['GAMMA']
        )
        #dataset = SupervisedLLCDataset(datapath)
        length = len(dataset)
        self._train_dataset_length = int(length * 0.75)
        self._val_dataset_length = length - int(length * 0.75)
        print('Training Data Size: {}'.format(str(self._train_dataset_length)))
        print('Validation Data Size: {}'.format(str(self._val_dataset_length)))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [int(length * 0.75), length - int(length * 0.75)],
            torch.Generator().manual_seed(42)
        )
        self._train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = params['batch_size'],
            shuffle = True
        )
        self._val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = params['batch_size'],
            shuffle = True
        )
        self._ep = 0
        self._epoch = 0
        self._n_step = 0
        self._prev_eval_loss = 1e8

    def learn(self, experiment):
        print('Start Training.')
        done = self._pretrain(experiment)
        if done:
            print('Training Done.')
        self._test()

    def _pretrain_step(self, x, y):
        loss = 0.0
        self._model.zero_grad()
        y_pred = self._model(x)
        loss += torch.nn.functional.mse_loss(y_pred, y)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        self.logger.add_scalar(
            'Train/Loss',
            loss.detach().cpu().numpy(),
            self._n_step
        )
        return loss.detach().cpu().numpy()

    def _pretrain_epoch(self):
        epoch_loss = 0.0
        self._step = 0
        self.logger.add_scalar('Train/Learning Rate', self._scheduler.get_last_lr()[0], self._epoch)
        for x, y in self._train_dataloader:
            # Modify the following line accordingly
            loss = self._pretrain_step_v4(x, y)
            epoch_loss += loss
            self._step += 1
            self._n_step += 1
            if self._step > params['max_epoch_size']:
                break
        if (self._epoch + 1) % params['scheduler_update_freq'] == 0:
            self._scheduler.step()
        return epoch_loss

    def _pretrain(self, experiment):
        """
            modify according to need
        """
        ep_loss = 0.0
        self._epoch = 0
        pbar = tqdm(total = params['n_epochs'])
        while self._epoch < params['n_epochs']:
            if ((self._ep + 1) * self._epoch) % params['n_eval_steps'] == 0:
                eval_loss = self._eval_v4()
                if eval_loss <= self._prev_eval_loss:
                    self._save(experiment)
                    self._prev_eval_loss = eval_loss
            epoch_loss = self._pretrain_epoch()
            self._epoch += 1
            if self._step > 0:
                epoch_loss = epoch_loss / self._step
            self.logger.add_scalar('Train/Epoch Loss', epoch_loss, self._epoch)
            pbar.update(1)
        pbar.close()
        return True

    def _save(self, experiment):
        torch.save(self._model, os.path.join(self.logdir, 'exp{}'.format(experiment),'controller.pth'))

    def _eval(self):
        step = 0
        loss = 0.0
        for x, y in self._val_dataloader:
            y_pred = self._model(x)
            loss += torch.nn.functional.mse_loss(y_pred, y)
            step += 1
        if step > 0:
            self.logger.add_scalar('Eval/Loss', loss.detach().cpu().numpy()/step, self._epoch)
            loss = loss.detach().cpu().numpy() / step
        else:
            raise AttributeError('Validation dataloader is empty')
        return loss

    def _load_model(self, path):
        print('Loading Model')
        self._model = torch.load(path, map_location = device)
        print('model loaded')