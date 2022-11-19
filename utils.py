import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import os
from glob import glob


class Trainer:
    def __init__(self, args, model, loader, edge_index, loss_fn, DEVICE):
        self.args = args
        self.model = model.to(DEVICE)
        self.train_loader, self.val_loader = loader
        self.loss_fn = loss_fn
        self.edge_index = edge_index.to(DEVICE)
        self.device = DEVICE
        self.min_val_loss = float('inf')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
    
    def train(self):
        loader_len = len(self.train_loader)
        print(f"\nTraining begin, {self.args.epochs} epochs, {loader_len} iters for each epoch.")
        self.model.train()
        for epoch in tqdm(range(self.args.epochs)):
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                # add Gaussian noise to the input
                if self.args.noise_std > 0:
                    x[:, :, 0, :] += torch.randn_like(x[:, :, 0, :])*self.args.noise_std
                y_hat = self.model(x, self.edge_index)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % 20 == 0:
                    self.writer.add_scalar('train/loss', loss.item(), epoch*loader_len + step)
            if epoch % 25 == 0:
                self.val(epoch)
        self.writer.close()
    
    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        val_loss = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x, self.edge_index)
            val_loss += self.loss_fn(y_hat, y)
        val_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar('val/loss', val_loss.item(), epoch)
        if val_loss < self.min_val_loss:
            self.model_save(name=f'Best_val_loss_{val_loss:.4f}')
            self.min_val_loss = val_loss
        self.model.train()

    def model_save(self, name):
        save_dir = self.args.log_dir
        for i in glob(f"{save_dir}/*.pth"):
            os.remove(i)
        torch.save(self.model.state_dict(), os.path.join(save_dir, name+'_model_state.pth'))


class Tester:
    def __init__(self, args, model, loader, edge_index, loss_fn, DEVICE):
        self.args = args
        self.model = model.to(DEVICE)
        self.test_loader = loader
        self.loss_fn = loss_fn
        self.edge_index = edge_index.to(DEVICE)
        self.device = DEVICE
        N = self.test_loader.dataset[0][1].shape[0]
        T = len(loader.dataset) + args.pre_time_len - 1
        self.predictions = np.zeros((N, T))
        self.gt = np.zeros((N, T))
        self.times = np.ones((1, T)) * args.pre_time_len
        for i in range(args.pre_time_len - 1):
            self.times[0, i] = i + 1
            self.times[0, -(i+1)] = i + 1

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.loss = 0
        for step, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x, self.edge_index)
            self.loss += self.loss_fn(y_hat, y)
            self.predictions[:, step:step+self.args.pre_time_len] += y_hat.squeeze().cpu().numpy()
            self.gt[:, step:step+self.args.pre_time_len] += y.squeeze().cpu().numpy()
        self.predictions /= self.times
        self.gt /= self.times
        self.loss /= len(self.test_loader)
        self.results_save()

    def results_save(self):
        np.save(os.path.join(self.args.log_dir, f'predictions_mse={self.loss:.4f}_.npy'), self.predictions)
        np.save(os.path.join(self.args.log_dir, f'gt.npy'), self.gt)
        print(f"Test results saved in {self.args.log_dir}")
