import torch
from utils import Controller, SupervisedLLCDataset, \
    ControllerV2, ControllerV3
import os
from constants import params
import shutil
import argparse
from torch.utils import tensorboard
from tqdm import tqdm

class Learner:
    def __init__(self, logdir, datapath, logger):
        self.logdir = logdir
        self.datapath = datapath
        self.logger = logger
        self._model = ControllerV3()
        self._optim  = torch.optim.Adam(
            self._model.parameters(),
            lr = params['LEARNING_RATE']
        ) 
        dataset = SupervisedLLCDataset(datapath)
        length = len(dataset)
        
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

    def learn(self, experiment):
        print('Start Training.')
        done = self._pretrain(experiment)
        if done:
            print('Training Done.')
    
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

    def _pretrain_step_v2(self, x, y):
        """
            This method is used to train the auto encoder architecture
        """
        loss = 0.0
        self._model.zero_grad()
        y_pred, x_pred = self._model(y)
        loss += torch.nn.functional.mse_loss(y_pred, y) + \
                torch.nn.functional.mse_loss(x_pred, x)
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
        for x, y in self._train_dataloader:
            # Modify the following line accordingly
            loss = self._pretrain_step_v2(x, y)
            epoch_loss += loss
            self._step += 1
            self._n_step += 1
        return epoch_loss

    def _pretrain(self, experiment):
        """ 
            modify according to need
        """
        ep_loss = 0.0 
        self._epoch = 0
        while self._epoch < params['n_epochs']:
            epoch_loss = self._pretrain_epoch()
            self._epoch += 1
            if self._step > 0:
                epoch_loss = epoch_loss / self._step
            self.logger.add_scalar('Train/Epoch Loss', epoch_loss, self._epoch)
            if (self._ep + 1 ) * self._epoch % params['n_eval_steps'] == 0:
                self._save(experiment)
                self._eval_v2()
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
        else:
            raise AttributeError('Validation dataloader is empty')

    def _eval_v2(self):
        step = 0 
        loss = 0.0 
        for x, y in self._val_dataloader:
            y_pred, x_pred = self._model(y)
            loss += torch.nn.functional.mse_loss(y_pred, y) + \
                    torch.nn.functional.mse_loss(x_pred, x)
            step += 1
        if step > 0:
            self.logger.add_scalar('Eval/Loss', loss.detach().cpu().numpy()/step, self._epoch)
        else:
            raise AttributeError('Validation dataloader is empty')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    args = parser.parse_args()
    datapath = 'assets/out/results'
    logdir = os.path.join(datapath, 'supervised_llc')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    else:
        shutil.rmtree(logdir)
        os.mkdir(logdir)
    if not os.path.exists(os.path.join(logdir,
        'exp{}'.format(args.experiment))):
        os.mkdir(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
    else:
        shutil(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
        os.mkdir(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
    logger = tensorboard.SummaryWriter(os.path.join(logdir, 'exp{}'.format(str(args.experiment)), 'tensorboard'))
    learner = Learner(logdir, datapath, logger)
    learner.learn(args.experiment)
