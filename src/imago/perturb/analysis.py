import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path

from imago.domains import summary_str
from imago.perturb import *
from imago.perturb.sampler import nm_search_simp, psearch
from imago.viz import *

"""
Toolkit for running multiple samples around a given perturbation direction, and accumulating
the deltas, and performing an analysis.

Analyses currently include:
- Counts of types of edits (heatmap)
"""


BGCOLOR=(200, 200, 200)


def scan(start_O, model, perturb:Perturb,
         render_fn, # Render function to use to save out
         sampler_fn=nm_search_simp,
         min_Odiff=1,
         num_samples=10, summary_dir="results/summary"):
    """
    For a given starting observation, sample sets of points corresponding
    to that perturbation, and generate a report as well as save out the
    identified directions.
    :param start_O:
    :return:
    """
    print("Scanning, perturb={}".format(str(perturb)))
    with torch.no_grad():
        summary_dir = Path(summary_dir)
        summary_dir.mkdir(exist_ok=True, parents=True)
        DEVICE = model.device
        start_Oh, Z_mu, Z_logvar, start_Z, start_Th = model(start_O)
        pt_analyzer = PointAnalyzer(model, perturb, start_Z)
        start_Z = ensure_numpy(start_Z)
        res_array = [start_Z]
        start_img = render_fn(start_Oh)
        start_img.save(Path(summary_dir, "start.png"))
        with open(Path(summary_dir, "summary.txt"), 'w') as f:
            f.write("Starting scene details:\n")
            f.write(summary_str(start_Th))
            f.write("\n")
            analysis_pts = []
            for step_idx in tqdm(range(num_samples)):
                end_Z = sampler_fn(start_Z, model, perturb, device=DEVICE)
                res_array.append(ensure_numpy(end_Z))
                print("Optim done, scanning")
                analysis_pt = pt_analyzer.scaled_analysis(end_Z, norm_dir=False, min_Odiff=min_Odiff)
                #analysis_pt = pt_analyzer.analyze(end_Z)
                if analysis_pt is None:
                    desc_str = "No Result"
                else:
                    analysis_pts.append(analysis_pt)
                    desc_str = str(analysis_pt)
                print("\n-----\n#{}\n{}\n".format(step_idx, desc_str))
                f.write("\n-----\n#{}\n{}\n".format(step_idx, desc_str))
                f.flush()
        npz_fpath = Path(summary_dir, "vecs.npz")
        np.savez(npz_fpath, *res_array)
        viz_analysis_pts(analysis_pts, pt_analyzer, render_fn, summary_dir)


def viz_analysis_pts(analysis_pts, pt_analyzer, render_fn, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    DEVICE = pt_analyzer.model.device
    start_Oh, start_Th = pt_analyzer.model.forward_Z(pt_analyzer.start_Z)
    start_img = render_fn(start_Oh, start_Th)
    start_img.save(Path(result_dir, "start.png"))
    def _img_tag(fpath):
        return """<img src="{}"/>""".format(fpath)
    with open(Path(result_dir, "index.html"), "w") as f:
        f.write("""<html><document>
        <pre>{start_desc}</pre>
        <table>
        <tr><td>Start</td><td>Step</td><td>Diff</td><td>Desc</td></tr>
        """.format(start_desc=summary_str(start_Th)))
        for idx, analysis_pt in enumerate(analysis_pts):
            analysis_step_dir = analysis_pt.get_direction()
            if analysis_step_dir is not None:
                step_Oh, step_Th = pt_analyzer.model.forward_Z(ensure_torch(DEVICE, analysis_step_dir))
                step_img = render_fn(step_Oh)
                step_img.save(Path(result_dir, "step_{:03d}.png".format(idx)))
                Odiff = step_Oh - start_Oh
                Odiff_img = render_fn(Odiff, step_Th)
                Odiff_img.save(Path(result_dir, "odiff_{:03d}.png".format(idx)))
                f.write("""<tr><td>{src_img}</td><td>{step_img}</td><td>{odiff_img}</td><td><pre>{desc_str}</pre></td></tr>""".format(
                    src_img=_img_tag("start.png"),
                    step_img=_img_tag("step_{:03d}.png".format(idx)),
                    odiff_img=_img_tag("odiff_{:03d}.png".format(idx)),
                    desc_str=str(analysis_pt)
                ))


def endpt_vs_starts(end_Z, start_Zs, D, model, render_fn,
                    results_dir, min_Odiff=0.1):
    """
    Given an end_Z direction and an inventory of start scenes,
    generate the full perturbation, and then a scaled one representing minimal
    changes in that direction to meet the perturbation criteria.
    Goal here is to identify if a given large magnitude direction has a similar
    pact for a range of starting scenes.

    :param start_Zs:
    :param end_Z:
    :return:
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir = Path(results_dir, "images")
    imgs_dir.mkdir(parents=True, exist_ok=True)
    DEVICE = model.device
    full_res = []
    scaled_res = []
    for idx, start_Z in tqdm(enumerate(start_Zs)):
        pa = PointAnalyzer(model, D, start_Z)
        full_pa = pa.analyze(end_Z)
        scaled_pa = pa.scaled_analysis(end_Z, verbose=False, min_Odiff=min_Odiff)
        full_res.append(full_pa)
        scaled_res.append(scaled_pa)
        start_Ohat, start_That = model.forward_Z(ensure_torch(DEVICE, start_Z))
        full_Ohat, full_That = model.forward_Z(ensure_torch(DEVICE,
                                                            end_Z))
        if scaled_pa is not None and scaled_pa.has_direction():
            scaled_Ohat, scaled_That = model.forward_Z(ensure_torch(DEVICE,
                                                                    scaled_pa.get_direction()).view((1, -1)))
        start_img = render_fn(start_Ohat, start_That,
                              notes="ORIG\n{}".format(summary_str(start_That)))
        start_img.save(Path(imgs_dir, "start_{}.png".format(idx)))
        full_img = render_fn(full_Ohat, full_That,
                             notes="DIFF\n{}".format(str(full_pa)))
        full_img.save(Path(imgs_dir, "full_{}.png".format(idx)))
        if scaled_pa is not None and scaled_pa.has_direction():
            scaled_img = render_fn(scaled_Ohat, scaled_That,
                                   notes="DIFF\n{}".format(str(scaled_pa)))
            scaled_img.save(Path(imgs_dir, "scaled_{}.png".format(idx)))

    def img_html(fpath, add_td=True):
        if add_td:
            return """<td><img src="{}"/></td>""".format(fpath)
        else:
            return """<img src="{}"/>""".format(fpath)

    with open(Path(results_dir, "index.html"), "w") as f:
        f.write("<html><body>")
        f.write("<table>")
        f.write("""<tr>
        <td><b>Start Scene</b></td>
        <td><b>Perturb Full Extent</b></td>
        <td><b>Perturb First Diff</b></td>
        </tr>\n""")
        for idx, (full, scaled) in enumerate(zip(full_res, scaled_res)):
            start_img_fpath = Path("images", "start_{}.png".format(idx))
            full_img_fpath = Path("images", "full_{}.png".format(idx))
            scaled_img_fpath = Path("images", "scaled_{}.png".format(idx))
            f.write("\t<tr>{}{}{}<tr>\n".format(
                img_html(start_img_fpath),
                img_html(full_img_fpath),
                img_html(scaled_img_fpath),
            ))
            print("- - - - -\nIDX {}\n".format(idx))
            print("Full:")
            print(str(full))
            print("\nScaled:")
            print(str(scaled))
        f.write("\n</table></body></html>")