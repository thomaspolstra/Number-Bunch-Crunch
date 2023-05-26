import os
import json
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model.VolatilityLSTM import VolatilityLSTM
from utils.gather_data import read_data
from utils.dataset_factory import create_dataloaders
from utils import file_utils

from typing import Tuple


class Experiment:
    def __init__(self, exp_name: str):
        config_data_path = os.path.join('configs', f'{exp_name}.json')
        # open config data
        with open(config_data_path, 'r') as file:
            self.config_data = json.load(file)

        # create saving directory
        self.name = self.config_data['name']
        self.save_dir = os.path.join('experiment_data', self.name)

        # create DataLoaders
        self.data = read_data(**self.config_data['dataset'])
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(self.data,
                                                                                  **self.config_data['loaders'])

        # get batch sizes
        self.n_train_batches = len(self.train_loader)
        self.n_val_batches = len(self.val_loader)
        self.n_test_batches = len(self.test_loader)

        # set up experiment
        self.n_epochs = self.config_data['experiment']['n_epochs']
        self.patience = self.config_data['experiment']['patience']
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_model = None

        # initialize model and optimizers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VolatilityLSTM(**self.config_data['model'])
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_data['experiment']['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

        self.init_model()

        self.load_experiment()

    def load_experiment(self):
        os.makedirs('experiment_data', exist_ok=True)

        if os.path.exists(self.save_dir):
            self.train_losses = file_utils.read_file_in_dir(self.save_dir, 'training_losses.txt')
            self.val_losses = file_utils.read_file_in_dir(self.save_dir, 'val_losses.txt')
            self.current_epoch = len(self.train_losses)

            state_dict = torch.load(os.path.join(self.save_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.save_dir)

    def init_model(self):
        if self.device == 'cuda':
            self.model = self.model.cuda().float()
            self.criterion = self.criterion.cuda()

    def run(self):
        best_loss = np.inf
        patience = 0
        print('Beginning Training')
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.n_epochs):
            start_time = datetime.now()
            self.current_epoch = epoch
            train_loss = self.train(epoch)
            val_loss = self.val(epoch)
            self.record_stats(train_loss, val_loss)
            self.log_epoch_stats(start_time)
            self.save_model()

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model = deepcopy(self.model)
            else:
                patience += 1
            if patience == self.patience:
                self.model = self.best_model
                print(f'Early stopping after {epoch} epochs')
                break

    def train(self, epoch: int) -> float:
        self.model.train()
        training_loss = 0

        for i, (rolling_window, targets) in enumerate(self.train_loader):
            rolling_window = rolling_window.to(device=self.device)
            targets = targets.to(device=self.device)

            self.optimizer.zero_grad()

            preds, _ = self.model(rolling_window)
            loss = self.criterion(preds[:, :-1], targets)
            loss.backward()

            self.optimizer.step()
            training_loss += loss.detach().item()

            if i % 1 == 0:
                self.print_stats('train', epoch, i, training_loss)

        self.scheduler.step()
        print('Finished Training Epoch')
        return training_loss / self.n_train_batches

    def val(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (rolling_window, targets) in enumerate(self.val_loader):
                rolling_window = rolling_window.to(device=self.device)
                targets = targets.to(device=self.device)

                preds, _ = self.model(rolling_window)
                loss = self.criterion(preds[:, :-1], targets)
                val_loss += loss.detach().item()

                if i % 1 == 0:
                    self.print_stats('val', epoch, i, val_loss)

        print('Finished Validation Epoch')
        return val_loss / self.n_val_batches

    def test(self) -> Tuple[float, float]:
        self.model.eval()
        test_loss_forced = 0
        test_loss_infer = 0

        with torch.no_grad():
            for i, (rolling_window, targets) in enumerate(self.test_loader):
                rolling_window = rolling_window.to(device=self.device)
                targets = targets.to(device=self.device)

                preds_forced, _ = self.model(rolling_window)
                preds_infer = self.model.predict(rolling_window, n_days=targets.size(1), keep_init=False)

                loss_forced = self.criterion(preds_forced[:, :-1], targets)
                loss_infer = self.criterion(preds_infer, targets)

                test_loss_forced += loss_forced.detach().item()
                test_loss_infer += loss_infer.detach().item()

                if i % 1 == 1:
                    self.print_stats('test', -1, i, test_loss_forced)

        result_str = f"Test Performance:\t" \
                     f"Teacher Forced Loss: {test_loss_forced / self.n_test_batches}\t" \
                     f"Unforced Loss: {test_loss_infer / self.n_test_batches}"

        self.log(result_str)

        return test_loss_forced / self.n_test_batches, test_loss_infer / self.n_test_batches

    def save_model(self):
        root_model_path = os.path.join(self.save_dir, 'latest_model.pt')
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def record_stats(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.plot_stats()

        file_utils.write_to_file_in_dir(self.save_dir, 'training_losses.txt', self.train_losses)
        file_utils.write_to_file_in_dir(self.save_dir, 'val_losses.txt', self.val_losses)

    def log(self, log_str):
        print(log_str)
        file_utils.log_to_file_in_dir(self.save_dir, 'all.log', log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.n_epochs - self.current_epoch - 1)
        train_loss = self.train_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        summary_str = f'Epoch: {self.current_epoch + 1}\t' \
                      f'Train Loss: {train_loss}\t' \
                      f'Val Loss: {val_loss}\t' \
                      f'Took: {time_elapsed}\t' \
                      f'ETA: {time_to_completion}'
        self.log(summary_str)

    def plot_stats(self):
        e = len(self.train_losses)
        x_axis = np.arange(1, e+1, 1)
        plt.figure()
        plt.plot(x_axis, self.train_losses, label='Training Loss')
        plt.plot(x_axis, self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.title(self.name + ' Stats Plot')
        plt.savefig(os.path.join(self.save_dir, 'stat_plot.png'))

    def print_stats(self, stat_type: str, epoch: int, iteration: int, loss: float):
        print(f'{stat_type.capitalize()}:\t'
              f'[{epoch}/{self.n_epochs}][{iteration}/{getattr(self, f"n_{stat_type}_batches")}]\t'
              f'Average Loss: {loss / (iteration + 1):.4f}')
