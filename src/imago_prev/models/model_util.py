import os
import torch
import errno
import re
import json
import shutil
from absl import logging
from pathlib import Path
from PIL import Image

def safe_mkdirs(dirpath):
    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(dirpath, exist_ok=True)
        return dirpath
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            print("Warning, mkdirs={}, err={}".format(dirpath, exc))
            pass


def _make_meta_fpath(fpath):
    """ Given a fpath to a checkpoint, gets the meta JSON for this corresponding checkpoint.
    """
    fpath = Path(fpath)        
    fpath_meta = os.path.join(fpath.parent, "{}.meta.json".format(fpath.stem))
    return fpath_meta


def _load_meta(fpath):
    """ Returns the meta JSOn at the given point, or returns None"""
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            meta_json = json.load(f)
            return meta_json
    return None


def _load_best_meta_loss(fpath):
    """ Given the checkpoint fpath, checks to see if the best exists.  If so,
    load its meta object and returns the loss value, otherwise returns None.
    """
    fpath = Path(fpath)
    best_meta_fpath = os.path.join(fpath.parent, "{}.best.meta.json".format(fpath.stem))
    best_meta_json = _load_meta(best_meta_fpath)
    if best_meta_json:
        return best_meta_json['loss']
    return None


def save_checkpoint(fpath, model, optimizer, step, loss=None):
    """ Saves the checkpoint, along with the current best run.  If Loss is given, then
    the system will keep track of the best loss, tracking it as fpath.stem + ".best.pt" and its
    meta as ".best.meta.json"
    """
    fpath = Path(fpath)
    fpath_meta = _make_meta_fpath(fpath)
    safe_mkdirs(fpath.parent)

    save_dict = {'state': model.state_dict(), 'step': step}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if loss is not None:
        save_dict['loss'] = loss
    torch.save(save_dict, str(fpath))
    
    meta_json = {
        'step': step,
        'loss': loss
    }
    with open(fpath_meta, 'w') as f:
        json.dump(meta_json, f, indent=2)
        
    # Check if the current loss beats the current best.
    best_meta_loss = _load_best_meta_loss(fpath)
    if (loss is not None):
        if (best_meta_loss is None) or (loss < best_meta_loss):
            # Update the best loss with the current
            dst = os.path.join(fpath.parent, "{}.best.pt".format(fpath.stem))
            logging.info("New Best checkpoint found, copying src={} to {}".format(fpath, dst))
            shutil.copyfile(fpath, dst)
            dst_meta = os.path.join(fpath.parent, "{}.best.meta.json".format(fpath.stem))
            with open(dst_meta, 'w') as f:
                json.dump(meta_json, f, indent=2)


def load_checkpoint(fname, model, optimizer, device=None):
    if device:
        checkpoint = torch.load(fname, map_location=device)
    else:
        checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    return step


def checkpoint_exists(fpath):
    return os.path.isfile(fpath)


def find_latest_checkpoint(model_dir):
    """ Convenience routines for identifying the latest files named '*_${EPOCH}.pt', where $EPOCH
    is specified as an integer.
    """
    lookup = {}
    max_epoch = -1
    for x in os.listdir(model_dir):
        m = re.search('.*_(\\d+).pt', x)
        if m is not None:
            epoch = int(m[1])
            lookup[epoch] = os.path.join(model_dir, x)
            if epoch > max_epoch:
                max_epoch = epoch
    if max_epoch > 0:
        return lookup[max_epoch]
    else:
        return None
    
    
def make_pil_grid(images, padding=2, nrow=8):
    """
    Given a list of images, generates corresponding grid.
    Presumes all images are the same width/height as the first image
    :param images: 
    :param nrows: 
    :return: 
    """
    total_imgs = len(images)
    total_rows = total_imgs // nrow
    total_width = sum([image.size[0] for image in images]) + len(images) * padding
    total_height = sum([image.size[1] for image in images]) + len(images) * padding
    tgt_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    x = padding
    y = padding
    for idx, image in enumerate(images):
        tgt_img.paste(image, (x, y))
        if (idx + 1) % nrow == 0:
            y += padding + image.size[1]
            x = 0
        else:
            x += padding + image.size[0]
    return tgt_img
    
    