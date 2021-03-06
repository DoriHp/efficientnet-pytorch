import os
import shutil
from abc import ABCMeta, abstractmethod
import math

import mlconfig
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from .metrics import Accuracy, Average
from torch.utils.tensorboard import SummaryWriter

class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_loader: data.DataLoader,
                 valid_loader: data.DataLoader, scheduler: optim.lr_scheduler._LRScheduler, device: torch.device,
                 log_dir: str, num_epochs: int, output_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Lr handler
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)

        self.epoch = 1
        self.best_acc = 0

    def fit(self):

        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        for self.epoch in epochs:

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()

            self.writer.add_scalar("train/loss", train_loss.value, self.epoch)
            self.writer.add_scalar("train/acc", train_acc.value, self.epoch)

            for param_group in self.optimizer.param_groups:
                self.writer.add_scalar("train/lr", float(param_group['lr']), self.epoch)

            self.writer.add_scalar("valid/loss", valid_loss.value, self.epoch)
            self.writer.add_scalar("valid/acc", valid_acc.value, self.epoch)

            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoint.pth'))
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}, '
                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                   f'best valid acc: {self.best_acc:.2f}')
            self.scheduler.step()

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        # Get length of dataloadet
        iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)

        train_loader = tqdm(self.train_loader, total=iterations, desc='Train', position=0, leave=True)
        for images, target in train_loader:
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(images)
            loss = F.cross_entropy(output, target)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss.update(loss.item(), number=images.size(0))
            train_acc.update(output, target)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        with open("log.txt", "a+") as f:
            f.seek(0)
            data = f.read(100)
            if len(data) > 0:
                f.write("\n")

            f.write("Training: loss %.5f \t acc %.5f" %(train_loss.value, train_acc.value))

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        iterations = math.ceil(len(self.valid_loader.dataset) / self.valid_loader.batch_size)

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, total=iterations, desc='Validate', position=0, leave=True)
            for images, target in valid_loader:
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.model(images)
                loss = F.cross_entropy(output, target)

                valid_loss.update(loss.item(), number=images.size(0))
                valid_acc.update(output, target)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

        with open("log.txt", "a+") as f:
            f.seek(0)
            data = f.read(100)
            if len(data) > 0:
                f.write("\n")

            f.write("Evaluation: loss %.5f \t acc %.5f" %(valid_loss.value, valid_acc.value))

        return valid_loss, valid_acc

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
