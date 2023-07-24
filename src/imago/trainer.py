import os
from absl import logging
from pathlib import Path
import json
import pdb
import numpy as np
import collections
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from imago.viz import render
from imago.domains import DictTensorDataset, OBS
from imago.model_utils import save_checkpoint, load_checkpoint, checkpoint_exists
from imago.utils import tile_images_grid, ensure_numpy

from .utils import ensure_torch_rec, stats
from .losses import model_loss, RECON, KL

def compute_beta(step, train_dataloader):
    """
    Computes the beta, basing it off the number
    of epochs from the dataloader

    TODO: Compute this based off of recon error
    :param step:
    :param train_dataloader:
    :return:
    """
    epoch_size = len(train_dataloader.dataset)
    if step <= epoch_size * 30:
        return 0
    elif step <= epoch_size * 50:
        return 1e-8
    elif step <= epoch_size * 60:
        return 1e-7
    elif step <= epoch_size * 70:
        return 1e-6
    elif step <= epoch_size * 90:
        return 1e-5
    elif step <= epoch_size * 110:
        return 1e-4
#    elif step <= 700000:
#        return 1e-3
#    elif step <= 800000:
#        return 1e-2
    return 1e-3  # Cap it at this point


def train(model, output_dir,
          mcbox_spec,
          train_dataloader,
          test_dataloader,
          render_fn=render,
          device="cpu", epochs=100,
          step_limit=None,
          lr=1e-3,
          W_recon=1.0,
          load_existing=True):
    """
    Runs a training loop over the given dataloaders.
    Note that this expects each DataSet getitem to conform to returning
    lists of tensors,

    (obs, value, actions)

    :param model:
    :param output_dir:
    :param mcbox_spec:
    :param train_dataloader:
    :param test_dataloader:
    :param device:
    :param epochs:
    :return:
    """
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    MODEL_FPATH = os.path.join(output_dir, "model.pt")
    tb_logdir = os.path.join(output_dir, "tb_logs")
    tb_sw = SummaryWriter(tb_logdir)
    logging.info("Saving TB logs to {}".format(tb_logdir))
    print("Saving TB logs to {}".format(tb_logdir))

    with open(Path(output_dir, "meta.json"), 'w') as f:
        params = {
            "lr": lr,
            "W_recon": W_recon,
            "device": device,
            "epochs" : epochs,
            "mcbox_spec": str(mcbox_spec),
            "output_dir": str(output_dir),
            "model_summary": str(model),
            "len train": len(train_dataloader.dataset),
            "len val": len(test_dataloader.dataset)
        }
        json.dump(params, f, indent=2)

    model.train()
    step = 0

    if checkpoint_exists(MODEL_FPATH):
        logging.info("Loading existing checkpoint from {}".format(MODEL_FPATH))
        print("Loading existing checkpoint from {}".format(MODEL_FPATH))        
        step = load_checkpoint(MODEL_FPATH, model, optimizer, device)
        logging.info("... resuming from step={}".format(step))
        print("... resuming from step={}".format(step))        

    def sample_imgs(use_test=True):
        model.eval()
        loader = train_dataloader
        prefix = "train"
        if use_test:
            prefix="test"
            loader = test_dataloader
        for datum in loader:
            if isinstance(datum, torch.Tensor):
                O = datum
            elif isinstance(datum, list):
                O = datum[0]
            elif isinstance(datum, dict):
                O = datum[OBS]
            else:
                raise Exception("Unsupported datum type={}".format(type(datum)))
            O = O[0:3]  # Limit to smaller subset
            gold_img = render_fn(O, mcbox_spec)
            Yh, Z_mu, Z_logvar, Z_logvar, Th = model(O, deterministic=False)
            summary_img = render_fn(Yh, mcbox_spec)
            gold_img.save(os.path.join(output_dir, "{}_gold.png".format(prefix)))
            summary_img.save(os.path.join(output_dir, "{}_recon.png".format(prefix)))
            break
        model.train()
        return gold_img, summary_img

    def validate(beta, step):
        model.eval()
        tqdm_iter = tqdm(test_dataloader)
        for datum in tqdm_iter:
            if isinstance(datum, torch.Tensor):
                O = datum
            elif isinstance(datum, list):
                O = datum[0]
                Ts = [D.to(device) for D in datum[1:]] # Additional targets beyond reconstruction targets                
            elif isinstance(datum, dict):
                O = datum[OBS]
                Ts = {k:v.to(device) for k, v in datum.items()
                      if k != OBS }
            else:
                raise Exception("Unsupported datum type={}".format(type(O)))
            O = O.to(device)
            loss, named_losses = model_loss(model, O, Ts=Ts, beta=beta)
            batch_size = O.shape[0]
            tb_sw.add_scalar("Val_Loss", loss.item() / batch_size, global_step=step)
            del O
            for loss_name, loss_val in named_losses.items():
                tb_sw.add_scalar("Val_{}".format(loss_name), loss_val / batch_size, global_step=step)
        model.train()

    for epoch in range(epochs):
        tqdm_iter = tqdm(train_dataloader)
        epoch_loss = 0.
        for datum in tqdm_iter:
            if isinstance(datum, torch.Tensor):
                O = datum
                Ts = {}
            elif isinstance(datum, list):
                O = datum[0]
                Ts = [D.to(device) for D in datum[1:]] # Additional targets beyond reconstruction targets
            elif isinstance(datum, dict):
                O = datum[OBS]
                Ts = {k:v.to(device) for k, v in datum.items()
                      if k != OBS }
            else:
                raise Exception("Unsupported datum type={}".format(type(datum)))

            optimizer.zero_grad()
            O = O.to(device)
            beta = compute_beta(step, train_dataloader)
            loss, named_losses = model_loss(model, O, Ts=Ts, beta=beta, W=W_recon)
            # Now generate losses for additional latent to targets
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_size = O.shape[0]
            step += batch_size
            epoch_num = int(step / len(train_dataloader.dataset))
            tqdm_iter.set_description("Epoch {} step={}, beta={:.5f}, loss={:.5f}, epoch loss={:5f}".format(
                epoch_num, step, beta, loss.item() / batch_size, epoch_loss))
            .add_scalar("Loss", loss.item() / batch_size, global_step=step)
            #tb_sw.add_scalar("KL Loss", named_losses[KL], global_step=step)
            #tb_sw.add_scalar("Recon Loss", named_losses[RECON], global_step=step)
            tb_sw.add_scalar("Beta", beta, global_step=step)
            for loss_name, loss_val in named_losses.items():
                tb_sw.add_scalar(loss_name, loss_val / batch_size, global_step=step)
        xform = transforms.ToTensor()            
        train_gold, train_recon = sample_imgs(use_test=False)
        train_summary = tile_images_grid((train_gold, train_recon), buffer=50, num_per_row=2)
        tb_sw.add_image("train_gold_recon", xform(train_summary), step)
        test_gold, test_recon = sample_imgs(use_test=True)
        test_summary = tile_images_grid((test_gold, test_recon), buffer=50, num_per_row=2)
        tb_sw.add_image("test_gold_recon", xform(test_summary), step)
        save_checkpoint(MODEL_FPATH, model, optimizer, step)
        validate(beta, step)
        if step_limit is not None and step >= step_limit:
            print("Encountered step limit, step={} >= {}, EXITTING TRAINING".format(step, step_limit))
            return


def eval(model, output_dir,
          mcbox_spec,
          train_dataloader,
          test_dataloader,
          device="cpu", epochs=100,
          load_existing=True):
    """
    Runs a training loop over the given dataloaders.
    Note that this expects each DataSet getitem to conform to returning
    lists of tensors,

    (obs, value, actions)

    :param model:
    :param output_dir:
    :param mcbox_spec:
    :param train_dataloader:
    :param test_dataloader:
    :param device:
    :param epochs:
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters())
    MODEL_FPATH = os.path.join(output_dir, "model.pt")
    tb_logdir = os.path.join(output_dir, "tb_logs")
    tb_sw = SummaryWriter(tb_logdir)
    logging.info("Saving TB logs to {}".format(tb_logdir))
    
    model.eval()

    logging.info("Loading existing checkpoint from {}".format(MODEL_FPATH))
    step = load_checkpoint(MODEL_FPATH, model, optimizer, device)
    beta = compute_beta(step, train_dataloader)
    logging.info("... trained to step={}".format(step))

    def validate(beta, step, dataloader):
        stats = collections.OrderedDict()
        def add_scalar(name, value):
            if name not in stats:
                stats[name] = []
            stats[name].append(value)
        tqdm_iter = tqdm(dataloader)
        for datum in tqdm_iter:
            if isinstance(datum, torch.Tensor):
                O = datum
            elif isinstance(datum, list):
                O = datum[0]
            else:
                raise Exception("Unsupported datum type={}".format(type(O)))
            Ts = [D.to(device) for D in datum[1:]]  # Additional targets beyond reconstruction targets
            O = O.to(device)
            loss, named_losses = model_loss(model, O, Ts=Ts, beta=beta)
            batch_size = O.shape[0]
            add_scalar("Loss", loss.item() / batch_size)
            for loss_name, loss_val in named_losses.items():
                add_scalar("{}".format(loss_name), loss_val / batch_size)
        return stats

    train_stats = validate(beta, step, train_dataloader)
    val_stats = validate(beta, step, test_dataloader)

    print("----------\nTRAIN\n")
    for k, vs in train_stats.items():
        print("\t{}:\t{}".format(k, stats(vs)))

    behav_macroavg = np.mean([np.mean(vs) for k, vs in train_stats.items() if k.startswith("B")])
    print("Behavior macro average={:.8f}".format(behav_macroavg))

    
    print("----------\nVAL\n")
    for k, vs in val_stats.items():
        print("\t{}:\t{}".format(k, stats(vs)))    

    behav_macroavg = np.mean([np.mean(vs) for k, vs in val_stats.items() if k.startswith("B")])
    print("Behavior macro average={:.8f}".format(behav_macroavg))
