import argparse
import copy
import sys
from accelerate.utils import set_seed
from torchvision.transforms.v2.functional import to_pil_image

from joker.prior.models import Joker2DPrior
from joker.utils import parse_config, config2argv, import_obj
from omegaconf import OmegaConf
from accelerate import Accelerator
from PIL import Image
import numpy as np
import glob
import torch
from tqdm.auto import tqdm
import os
from torch.utils.data import default_collate
from pathlib import Path


# from evaluation.evaluate import evaluate_folder
# from torchvision.transforms.functional import to_pil_image
# from fastcomposer.data import collate_fn
# from fastcomposer.model import FastComposerModel
# from fastcomposer.data import DiffPortrait3DDataset
# from fastcomposer.data import OODDataset
# from fastcomposer.utils import import_obj


@torch.no_grad()
def inference_on_dataset(model, dataset, output_dir, accelerator=None, neval=-1, seed_from_img_prefix=False, **inference_kwargs):
    accelerator = Accelerator() if accelerator is None else accelerator

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    neval = min(len(dataset), neval) if neval > 0 else len(dataset)
    all_idcs = np.arange(neval)
    my_idcs = np.array_split(all_idcs, accelerator.num_processes)[accelerator.process_index]

    case_progress_bar = tqdm(my_idcs, disable=not accelerator.is_local_main_process, desc="Rendering Facescape Samples")
    for case_id in case_progress_bar:
        sample = dataset[case_id]
        batch = default_collate([sample])
        if seed_from_img_prefix:
            generator = torch.Generator(accelerator.device).manual_seed(int(dataset.get_sample_file_paths(case_id)["target_image_path"].stem.split("_")[0]))
        else:
            generator = torch.Generator(accelerator.device).manual_seed(int(case_id))
        output_images = model.infer(
            prompt=batch["caption"],
            control_map=batch["normal"],  # Nx3xHxW, -1 ... +1
            ref_imgs=batch["ref_imgs"],  # Nx1x3xHxW, 0 ... 1
            generator=generator,
            **inference_kwargs,
        )

        out_pred_name = os.path.join(output_dir, f"{case_id:06d}_pred.png")
        out_gt_name = os.path.join(output_dir, f"{case_id:06d}_gt.png")
        out_prompt_name = os.path.join(output_dir, f"{case_id:06d}_prompt.txt")
        out_mask_name = os.path.join(output_dir, f"{case_id:06d}_mask.png")
        out_ref_name = os.path.join(output_dir, f"{case_id:06d}_ref.png")
        out_ctrl_name = os.path.join(output_dir, f"{case_id:06d}_ctrl.png")
        to_pil_image(output_images[0], mode="RGB").save(out_pred_name)
        to_pil_image(batch["img"][0]).save(out_gt_name)
        to_pil_image(batch["mask"][0]).save(out_mask_name)
        to_pil_image(batch["normal"][0] * .5 + .5).save(out_ctrl_name)
        to_pil_image(batch["ref_imgs"][0, 0]).save(out_ref_name)
        with open(out_prompt_name, "w") as f:
            f.write(batch["caption"][0])

    return


if __name__ == "__main__":
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('10.1.5.155', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    config = OmegaConf.load(sys.argv[1])
    OmegaConf.resolve(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Joker2DPrior.from_pretrained()
    if config.load_model is not None:
        model.load_state_dict(
            torch.load(config.load_model, map_location="cpu")
        )  # watch out: potentially overwrites corrected vae!
    model.to(device)

    val_set_cfg = config.val_dataset
    dataset = import_obj(val_set_cfg._target)(**val_set_cfg._kwargs)

    model.eval()
    inference_on_dataset(model, dataset, config.output_dir, seed_from_img_prefix=True)
