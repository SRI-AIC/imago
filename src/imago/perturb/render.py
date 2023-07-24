"""
Rendering utilities for examining perturbations
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from imago.viz2 import render_stacked
from imago.utils import *

def viz_rollout(start_Z, end_Z,
                get_Z_fn,
                mcbox_spec, palettes,
                steps=20, 
                device="cpu"):
    start_Z = ensure_torch(device, start_Z)
    end_Z = ensure_torch(device, end_Z)
    dZ = (end_Z - start_Z) / steps
    
    interp_images = []
    with torch.no_grad():
        for i in range(0, steps + 1):
            Zp = start_Z + i * dZ 
            Ohat, That, entr_pA, iV = get_Z_fn(Zp)
            if i == 0:
                Ostart = Ohat
            desc = "Step {}, V={:.3f}, EntA={:.5f}".format(i, iV[0], entr_pA[0])
             # recon_img = viz.render(Ohat, mcbox_spec, cell_width=15)
            recon_img_stacked = render_stacked(Ohat, mcbox_spec, palettes=palettes, cell_width=15)
            
            # Draw the header
            header_img = PIL.Image.new("RGB", (recon_img_stacked.width, 40), (50,50,125))
            header_drawer = PIL.ImageDraw.Draw(header_img)
            try:
                label_font = PIL.ImageFont.truetype("arial.ttf", 12)
            except OSError:
                label_font = PIL.ImageFont.load_default()
            header_drawer.text((10, 20), desc, font=label_font, fill=(255,255,255))

            interp_images.append(tile_images_vert([header_img, recon_img_stacked]))
    delta_O = torch.norm(Ohat - Ostart)
    return tile_images_horiz(interp_images), delta_O

