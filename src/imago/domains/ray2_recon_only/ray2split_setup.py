"""
Implements the split model, which trains a recon first and then the
outcome on the recon's latent.
"""
from pathlib import Path
from imago.domains.reaverassaulty2.ray2_setup import *
from imago import MODELS_ROOT

SPLIT_MODEL_FPATH = Path(MODELS_ROOT, "ray2_recon_only", "outcome4latent", "model.pt")

def load_domain(device="cpu"):
    model = load_model(model_path=SPLIT_MODEL_FPATH, device=device)
    return ImagoDomain(model, compute_Odiff, render_ray2)