"""
Converts Trajectory Datasets into .sprite format used by the Disentangled Sequential Autoencoder project.
"""

import os
import torch
import torchvision.transforms as transforms

slice_transform = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def save_as_sprites(traj_ds, target_dir, renderer, num_frames=8):
    """ Save as .sprite file, for Sequential DisentangledVAE model"""
    os.makedirs(target_dir, exist_ok=True)
    idx = 1
    for (X, Y, C) in traj_ds.trajectories:
        if X.shape[0] >= num_frames:
            img = renderer.display(X[0:num_frames])
            sprites = slice_transform(img)
            torch.save(sprites, os.path.join(target_dir, "{}.sprite".format(idx)))
            idx += 1
