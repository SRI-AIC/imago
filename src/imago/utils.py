import numpy
import torch
import numpy as np
import PIL
import math
import random
import itertools

def flatten(regular_list):
    return list(itertools.chain(*regular_list))


def ensure_item(X):
    # Get a single scalar value
    if isinstance(X, float):
        return X
    else:
        return ensure_numpy(X).flatten()[0]


def ensure_torch_rec(device, X):
    if isinstance(X, torch.Tensor):
        return X.to(device)
    elif isinstance(X, np.ndarray):
        return torch.from_numpy(X.astype(np.float32)).to(device)
        #return torch.Tensor(X, dtype=torch.float32).to(device)
    elif isinstance(X, list):
        return [ensure_torch_rec(device, x) for x in X]
    else:
        raise TypeError("Unknown type={}".format(type(X)))


def ensure_torch(device, X):
    return ensure_torch_rec(device, X)


def tile_images_grid(images, buffer=10, num_per_row=5, bgcolor=(0,0,0)):
    widths, heights = zip(*(i.size for i in images))
    max_height = max(heights)
    max_width = max(widths)
    total_width = num_per_row * (max_width + buffer)
    total_height = math.ceil(len(images) / num_per_row) * (max_height + buffer)
    new_im = PIL.Image.new('RGB', (total_width, total_height), color=bgcolor)
    x_offset = 0
    y_offset = 0
    for idx, im in enumerate(images):
        if idx > 0 and idx % num_per_row == 0:
            x_offset = 0
            y_offset += max_height + buffer
        assert y_offset < total_height
        assert x_offset < total_width
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0] + buffer
    return new_im

def tile_images_horiz(images, buffer=10):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + buffer*len(widths)
    max_height = max(heights)
    new_im = PIL.Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] + buffer
    return new_im

def tile_images_vert(images, buffer=10):
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights) + buffer*len(heights)
    max_width = max(widths)
    new_im = PIL.Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + buffer
    return new_im

def set_random_seed(SEED=1337):
    """ Unified random seed generator
    """
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def ensure_numpy(tensor):
    """ If the tensor is a torch one, detaches and converts to a numpy array.
    Otherwise returns as-is (implicitly a numpy structure).
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        # TODO: Ensure this is actually a valid numpy structure
        return tensor


def sigmoid(X):
    if isinstance(X, torch.Tensor):
        return torch.nn.functional.sigmoid(X)
    else:
        return 1. / (1. + np.exp(-X))


def scan_stats(X):
    mu, std, xmin, xmax = np.mean(X), np.std(X), np.min(X), np.max(X)
    return mu, std, xmin, xmax

def stats(X):
    mu, std, xmin, xmax = scan_stats(X)
    return "mu/std={:.8f}/{:.8f}, min/max={:.8f}/{:.8f}".format(mu, std, xmin, xmax)


def bracketed(a, b, x):
    """
    Returns if x is inside bounds described by a, b
    """
    if a < b:
        return a <= x <= b
    else:
        return b <= x < a


def scan(X):
    mu, std, xmin, xmax = np.mean(X), np.std(X), np.min(X), np.max(X)
    return "mu/std={:.3f}/{:.3f}, min/max={:.3f}/{:.3f}".format(mu, std, xmin, xmax)