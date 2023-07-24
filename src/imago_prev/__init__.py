import torch
import os
import pathlib

# Imago project root directory
IMAGO_ROOT = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent

def logits2label(X, dim=0):
    """
    Given CxHxW array, converts into HxW where values are softmax offsets
    """
    return torch.argmax(X, dim=0)
