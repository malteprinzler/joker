"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import copy
import json

# import pydevd_pycharm
# pydevd_pycharm.settrace('10.34.31.195', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno.datasets.ImageTestDataset import TestData, CelebAHQData, NerSembleData
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
import matplotlib.pyplot as plt


def test(model, batch):
    batch["image"] = batch["image"].cuda()
    if len(batch["image"].shape) == 3:
        batch["image"] = batch["image"].view(1, 3, 224, 224)
    values = model(batch, training=False, validation=False)
    return values


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    path_to_models = config.kwargs.path_to_models
    input_folder = config.kwargs.input_folder
    output_folder = config.kwargs.output_folder
    model_name = config.kwargs.model_name
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {args.config} {output_folder}/config.yaml")

    if config.kwargs.seed is not None:
        seed_everything(config.kwargs.seed)

    # 1) Load the model
    face_rec_model, conf = load_model(path_to_models, model_name)
    face_rec_model.cuda()
    face_rec_model.eval()

    # 2) Create a dataset
    dataset = NerSembleData(input_folder, face_detector="fan", max_detection=1, frame_file=config.kwargs.get("frame_file", None))

    ## 4) Run the model on the data
    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    idcs = np.array_split(np.arange(len(dataset)), njobs)[jobid]
    for i in auto.tqdm(idcs):
        batch = dataset[i]
        if "image" in batch:
            try:
                vals = test(face_rec_model, batch)
                visdict = face_rec_model.visualize_batch(batch, i, None, in_batch_idx=None)
                current_bs = vals["shapecode"].shape[0]
                for k, v in vals.items():
                    if isinstance(v, torch.Tensor):
                        vals[k] = v.cpu().tolist()
                    elif isinstance(v, np.ndarray):
                        vals[k] = v.tolist()
            except Exception as e:
                print(f"ERROR with file {batch['image_path']}: {e}")
                H, W = batch["image_original"].shape[-2:]
                H_uv = W_uv = face_rec_model.renderer.render.uv_size
                vals = dict(
                    shapecode=[None],
                    texcode=[None],
                    jawpose=[None],
                    globalpose=[None],
                    cam=[None],
                    cam_out=[None],
                    tform=[None],
                    H_orig=[None],
                    H_crop=[None],
                    lightcode=[None],
                    expcode=[None],
                )
                visdict = dict(
                    shape_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                    normal_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                    uv_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                    uvinv_image=np.zeros((1, H_uv, W_uv, 3), dtype=np.uint8),

                )
                current_bs = 1

        else:
            H, W = batch["image_original"].shape[-2:]
            H_uv = W_uv = face_rec_model.renderer.render.uv_size
            vals = dict(
                shapecode=[None],
                texcode=[None],
                jawpose=[None],
                globalpose=[None],
                cam=[None],
                cam_out=[None],
                tform=[None],
                H_orig=[None],
                H_crop=[None],
                lightcode=[None],
                expcode=[None],
            )
            visdict = dict(
                shape_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                normal_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                uv_image=np.zeros((1, H, W, 3), dtype=np.uint8),
                uvinv_image=np.zeros((1, H_uv, W_uv, 3), dtype=np.uint8),

            )
            current_bs = 1

        for j in range(current_bs):
            rel_src_path = Path(batch["image_path"][0]).relative_to(input_folder)
            src_dir_name = rel_src_path.parent.name
            src_stem = rel_src_path.stem
            out_dir = Path(output_folder) / rel_src_path.parents[1]

            outpath_params = out_dir / src_dir_name.replace("images", "flame_params") / (src_stem + ".json")
            outpath_shape = out_dir / src_dir_name.replace("images", "flame_shape") / (src_stem + ".jpg")
            outpath_normal = out_dir / src_dir_name.replace("images", "flame_normals") / (src_stem + ".jpg")
            outpath_uv = out_dir / src_dir_name.replace("images", "flame_uvs") / (src_stem + ".png")
            outpath_uvinv = out_dir / src_dir_name.replace("images", "flame_uvinvs") / (src_stem + ".png")

            outpath_params.parent.mkdir(exist_ok=True, parents=True)
            outpath_shape.parent.mkdir(exist_ok=True, parents=True)
            outpath_normal.parent.mkdir(exist_ok=True, parents=True)
            outpath_uv.parent.mkdir(exist_ok=True, parents=True)
            outpath_uvinv.parent.mkdir(exist_ok=True, parents=True)

            with open(outpath_params, "w") as f:
                json.dump(
                    dict(
                        shapecode=vals["shapecode"][j],
                        texcode=vals["texcode"][j],
                        jawpose=vals["jawpose"][j],
                        globalpose=vals["globalpose"][j],
                        cam=vals["cam"][j],
                        cam_out=vals["cam_out"][j],
                        tform=vals["tform"][j],
                        H_orig=vals["H_orig"][j],
                        H_crop=vals["H_crop"][j],
                        lightcode=vals["lightcode"][j],
                        expcode=vals["expcode"][j],
                    ),
                    f,
                    indent="\t")
            imsave(outpath_shape, visdict["shape_image"][j])
            imsave(outpath_normal, visdict["normal_image"][j])
            imsave(outpath_uv, visdict["uv_image"][j])
            imsave(outpath_uvinv, visdict["uvinv_image"][j])

    print("Done")


if __name__ == '__main__':
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('10.34.31.195', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    main()
