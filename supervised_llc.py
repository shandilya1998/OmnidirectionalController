import torch
from utils import Controller, SupervisedLLCDataset, \
    ControllerV2, ControllerV3, ControllerV4, ControllerV5
import os
from constants import params
import shutil
import argparse
from torch.utils import tensorboard
from tqdm import tqdm
from simulations import QuadrupedV3
import numpy as np

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
        dataset = SupervisedLLCDataset(datapath)
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

    def _pretrain_step_v3(self, x, y):
        """
            This method is used to train the auto encoder architecture
        """
        loss = 0.0
        self._model.zero_grad()
        y_pred, z, z_pred  = self._model(y, x)
        loss += torch.nn.functional.mse_loss(y_pred, y) + \
                torch.nn.functional.mse_loss(z_pred, z)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        self.logger.add_scalar(
            'Train/Loss',
            loss.detach().cpu().numpy(),
            self._n_step
        )
        return loss.detach().cpu().numpy()

    def _pretrain_step_v4(self, x, y):
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
        self._test()
        return loss

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
            loss = loss.detach().cpu().numpy() / step
        else:
            raise AttributeError('Validation dataloader is empty')
        self._test()
        return loss

    def _eval_v3(self):
        step = 0
        loss = 0.0
        for x, y in self._val_dataloader:
            y_pred, z, z_pred  = self._model(y, x)
            loss += torch.nn.functional.mse_loss(y_pred, y) + \
                    torch.nn.functional.mse_loss(z_pred, z)
            step += 1
        if step > 0:
            self.logger.add_scalar('Eval/Loss', loss.detach().cpu().numpy()/step, self._epoch)
            loss = loss.detach().cpu().numpy() / step
        else:
            raise AttributeError('Validation dataloader is empty')
        self._eval_llc()
        self._test()
        return loss

    def _eval_v4(self):
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

    def _eval_llc(self):
        step = 0
        loss = 0.0
        for x, y in self._val_dataloader:
            y_pred = self._model.decoder(self._model.transform(x))
            loss += torch.nn.functional.mse_loss(y_pred, y)
            step += 1
        if step > 0:
            self.logger.add_scalar('Eval/Control Error', loss.detach().cpu().numpy()/step, self._epoch)
            loss = loss.detach().cpu().numpy() / step
        else:
            raise AttributeError('Validation dataloader is empty')
        return loss

    def _load_model(self, path):
        print('Loading Model')
        self._model = torch.load(path)
        self._model = Controller()
        self._optim  = torch.optim.Adam(
            self._model.parameters(),
            lr = params['LEARNING_RATE']
        )
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optim,
            gamma = params['GAMMA']
        )

    def _test(self):
        env = QuadrupedV3()
        ob = env.reset()
        steps = 0
        X = np.load(os.path.join(self.datapath, 'X.npy'))
        index = np.random.randint(low = 0, high = X.shape[0])
        env._set_goal(X[index][:6])
        while steps < params['MAX_STEPS']:
            x = torch.from_numpy(np.expand_dims(np.concatenate([
                ob['desired_goal'],
                ob['achieved_goal'],
                ob['observation']
            ], -1), 0).astype('float32'))
            y = self._model(x).detach().cpu().numpy()[0]
            ob, reward, done, info = env.step(y)
            env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--test',
        nargs='?', type = int, const = 1,
        help = 'choice to use script in test or training mode'
    )
    args = parser.parse_args()
    datapath = 'assets/out/results_v2'
    logdir = os.path.join(datapath, 'supervised_llc')
    logger = tensorboard.SummaryWriter(os.path.join(logdir, 'exp{}'.format(str(args.experiment)), 'tensorboard'))
    if args.test is None:
        datapath = 'assets/out/results_v2'
        logdir = os.path.join(datapath, 'supervised_llc')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if not os.path.exists(os.path.join(logdir,
            'exp{}'.format(args.experiment))):
            os.mkdir(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
        else:
            shutil.rmtree(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
            os.mkdir(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
        learner = Learner(logdir, datapath, logger)
        learner.learn(args.experiment)
    else:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if not os.path.exists(os.path.join(logdir,
            'exp{}'.format(args.experiment))):
            os.mkdir(os.path.join(logdir, 'exp{}'.format(str(args.experiment))))
        learner = Learner(logdir, datapath, logger)
        learner._load_model(os.path.join(logdir, 'exp{}'.format(args.experiment),'controller.pth'))
        learner._test()
