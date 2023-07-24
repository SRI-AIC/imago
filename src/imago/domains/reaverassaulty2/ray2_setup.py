import os
from pathlib import Path
import moviepy.editor as mp

from multiprocessing.pool import Pool
from imago_prev.data.datasets import IDPDataset

from imago import PVAE_ROOT, IMAGO_ROOT, CAML_ROOT
from imago.viz import render
from imago.models.model import PVaeModel
from imago.spaces.mc_box import *
from gym.spaces.box import Box

from imago.domains import summary_str, var_shortname, VALUE_FUNCTION, CONFIDENCE, GOAL_COND, RISKINESS, INCONGRUITY, OBS
from imago.domains import ImagoDomain
from imago.spaces.mc_box import MCBoxSpec
from imago.spaces.channels import SPACE_TYPE, ChannelSpec
from imago.domains.reaverassaulty2.ray2_data import load_data
from imago.model_utils import load_checkpoint, checkpoint_exists

TUNED_ODIFF_THRESH = 107  # Tuned on March 31st, 2022

shortname2var = {v:k for k, v in var_shortname.items()}

obs_space = Box(0, 2048, shape=(3, 64, 64), dtype=np.uint8)
OBS_CHANNEL_SPECS = [
    ChannelSpec("player_relative", SPACE_TYPE.CATEGORICAL, low=0, high=4),
    ChannelSpec("unit_type", SPACE_TYPE.CATEGORICAL, low=0, high=2048),
    ChannelSpec("unit_hit_points_ratio", SPACE_TYPE.SPARSE_NUMERIC, low=0, high=255, apply_whitening=True)
]
mcbox_spec = MCBoxSpec(OBS_CHANNEL_SPECS, obs_space, channel_axis=0)

LATENT2TARGET_SPECS = [
    ChannelSpec(VALUE_FUNCTION, SPACE_TYPE.NUMERIC, shape=(1,), low=-1000, high=6000,
                mean=219.058, std=694.972,
                apply_whitening=False, apply_recenter=True),
    ChannelSpec(CONFIDENCE, SPACE_TYPE.NUMERIC, shape=(1,), low=0, high=1, apply_whitening=True),
    ChannelSpec(GOAL_COND, SPACE_TYPE.NUMERIC, shape=(1,), low=-1, high=1, apply_whitening=True),
    ChannelSpec(RISKINESS, SPACE_TYPE.NUMERIC, shape=(1,), low=-5, high=5, apply_whitening=True),
    ChannelSpec(INCONGRUITY, SPACE_TYPE.NUMERIC, shape=(1,), low=-1, high=1, apply_whitening=True)
]

DEFAULT_MODEL_FPATH = Path(IMAGO_ROOT, "models", "ReaverAssaultYear2", "model.pt")
DATA_ROOT = Path(CAML_ROOT, "datasets", "ReaverAssaultYear2")
DEFAULT_CSV_FPATH = Path(DATA_ROOT, "interestingness.csv.gz")


def load_model(model_path=DEFAULT_MODEL_FPATH, device="cpu", force_new=False):
    model = PVaeModel(mcbox_spec, latent2target_specs=LATENT2TARGET_SPECS,
                      device=device).to(device)
    if checkpoint_exists(model_path):
        print("Checkpoint found at model_path={}".format(model_path))
    else:
        print("No checkpoint at specified model_path={}".format(model_path))
    if not(force_new) and checkpoint_exists(model_path):
        print("Loading from checkpoint at {}".format(model_path))
        load_checkpoint(model_path, model, None, device=device)
    else:
        print("Instancing new model")
    return model


def load_domain(device="cpu"):
    model = load_model(device=device)
    return ImagoDomain(model, compute_Odiff, render_ray2)


def render_ray2(O, That=None):
    if len(O.shape) == 3:
        O = np.expand_dims(O, axis=0)
    if isinstance(That, dict):
        That = [That]
    if That is not None:
        notes_arr = [summary_str(that) for that in That]
        return render(O, space_specifier=mcbox_spec, notes_arr=notes_arr)
    else:
        return render(O, space_specifier=mcbox_spec)


def compute_Odiff(O1, O2, default_device="cuda"):
    # Needs batch indices to be present
    assert len(O1.shape) == 4 and len(O2.shape) == 4
    O1 = ensure_torch(default_device, O1)
    O2 = ensure_torch(default_device, O2)
    O1 = mcbox_spec.make_onehot(O1, device=O1.device)
    O2 = mcbox_spec.make_onehot(O2, device=O1.device)
    return torch.sum(torch.abs(O1 - O2)).item()


def save_ep_movie(ds, ep, fpath,
                  time_per_frame=0.2,
                  model=None):
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    idxes = ds.get_ep_idxes(ep)
    imgs = []
    z_imgs = []
    duration = len(idxes) * time_per_frame
    animated_gif_fpath = str(fpath) + ".gif"
    frame_num = 0
    for idx in idxes:
        datum = ds[idx]
        O = datum[OBS]
        datum["_frame"] = np.array(frame_num)
        img = render_ray2(O, datum)
        imgs.append(img)
        if model is not None:
            O = ensure_torch(model.device, O).unsqueeze(0)
            Ohat, _, _, _, That = model(O, ret_for_training=False)
            That["_frame"] = np.array(frame_num)
            z_img = render_ray2(Ohat, [That])
            z_imgs.append(z_img)
        frame_num += 1
    imgs[0].save(animated_gif_fpath, save_all=True, append_images=imgs[1:],
                                duration=duration, loop=0)
    clip = mp.VideoFileClip(animated_gif_fpath)
    clip.write_videofile(str(fpath))
    os.remove(animated_gif_fpath)
    if model is not None:
        z_imgs[0].save(animated_gif_fpath, save_all=True,
                                      append_images=z_imgs[1:],
                                      duration=duration, loop=0)
        clip = mp.VideoFileClip(animated_gif_fpath)
        z_fpath = Path(fpath.parent, "{}.z.mp4".format(fpath.stem))
        clip.write_videofile(str(z_fpath))