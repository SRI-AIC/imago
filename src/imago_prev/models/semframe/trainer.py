from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

from imago_prev.models.model_util import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from imago_prev.models.model_util import save_checkpoint, load_checkpoint, checkpoint_exists, make_pil_grid
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from absl import logging, flags

from .frames import *
from .plot import render_frames
from torchvision.transforms import ToTensor
from .frames import CategoryLookup, make_category_lookup

        
        
class SC2VAETrainer:
    def __init__(self, root_dir, model, model_name="sc2vae", resume=True, LR=1e-3, device="cpu", beta=1,
                deterministic=False):
        self.root_dir = root_dir
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.device = device
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(root_dir, "checkpoints")
        self.steps = 0
        self.beta = beta
        logging.info("Beta={}".format(self.beta))
        self.deterministic = deterministic
        if deterministic:
            logging.info("Deterministic is TRUE, will act as autoencoder!")
        self.checkpoint_fpath = os.path.join(self.checkpoint_dir, self.model_name+".pt")
        logging.info("Checking fpath={}".format(self.checkpoint_fpath))
        if os.path.isfile(self.checkpoint_fpath) and resume:
            logging.info("Resuming from checkpoint = {}".format(self.checkpoint_fpath))
            self.steps = load_checkpoint(self.checkpoint_fpath, self.model, self.optimizer,
                            device=self.device)
            logging.info("Resuming from step = {}".format(self.steps))
        os.makedirs(self.root_dir, exist_ok=True)
        self.tb_logdir = os.path.join(self.root_dir, "tb_logs")
        self.tb_sw = SummaryWriter(self.tb_logdir)
        
    def train(self, train_dataloader, val_dataloader, epochs=10, save_interval=10000):
        save_countdown = save_interval
        counter = 0

        def _render(prefix=""):
            # Sample images, train and validation
            xforms = ToTensor()

            total = len(train_dataloader.dataset)
            vidxes = [3, total // 4, (3 * total) // 5]
            for vidx in vidxes:
                train_obs_batch = train_dataloader.dataset.get_batch([vidx])
                train_res = self.model.fwd2dict(train_obs_batch, deterministic=self.deterministic)
                train_img = render_frames(train_res[0], self.model.components)
                train_gold_img = render_frames(train_obs_batch[0], self.model.components)
                combined_img = make_pil_grid([train_img, train_gold_img], padding=10, nrow=1)
                combined_img = xforms(combined_img)
                self.tb_sw.add_image("{}TRAIN guess and gold render #{}".format(prefix, vidx), combined_img, self.steps)

            total = len(val_dataloader.dataset)
            vidxes = [3, total // 4, (3 * total) // 5]
            for vidx in vidxes:
                val_obs_batch = val_dataloader.dataset.get_batch([vidx])
                val_res = self.model.fwd2dict(val_obs_batch, deterministic=self.deterministic)
                val_img = render_frames(val_res[0], self.model.components)
                val_gold_img = render_frames(val_obs_batch[0], self.model.components)
                combined_img = make_pil_grid([val_img, val_gold_img], padding=10, nrow=1)
                combined_img = xforms(combined_img)
                self.tb_sw.add_image("{}VAL guess and gold render #{}".format(prefix, vidx), combined_img, self.steps)
                
        def _serialize():
            logging.info("Saving checkpoint to {}".format(self.checkpoint_fpath))
            save_checkpoint(self.checkpoint_fpath, model=self.model, optimizer=self.optimizer, step=self.steps)
            _render()

        # _render("START ")
            
        for epoch in range(epochs):
            #train_iter = tqdm(train_dataloader)
            epoch_loss = 0.
            for datums in tqdm(train_dataloader):
                O = datums[0]
                Ohat, z_mu, z_logvar, Z = self.model.forward(O)
                self.optimizer.zero_grad()
                loss = self.model.sc2_mf_out.loss_fn(Ohat, O)
                recon_loss = loss.item()
                # Add KL Loss for Posteriors
                kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1))
                loss += self.beta * kl_loss
                train_loss = loss.item()
                
                if counter % 5 == 0:
                    self.tb_sw.add_scalar("Recon_Loss",recon_loss/ len(obs_batch_indices), self.steps)
                    self.tb_sw.add_scalar("Raw_KL_Loss",kl_loss/ len(obs_batch_indices), self.steps)
                    self.tb_sw.add_scalar("Loss", train_loss / len(obs_batch_indices), self.steps)

                epoch_loss += train_loss
                loss.backward()
                self.optimizer.step()
                #train_iter.set_description("Epoch{}, Step={}\tLoss={:.5f}\tEpoch Loss={:.5f}".format(epoch, self.steps, train_loss, epoch_loss))
                self.steps += train_dataloader.batch_size
                save_countdown -= train_dataloader.batch_size
                counter += 1
                if save_countdown <= 0:
                    _serialize()
                    save_countdown = save_interval
            logging.info("Epoch {} Loss={:.5f}".format(epoch, epoch_loss))
        _serialize()

            
        
        
    