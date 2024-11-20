import copy

import PIL.Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from accelerate import Accelerator
from typing import List
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch
import glob
import os
import torchmetrics
from insightface.app import FaceAnalysis
from pandas import DataFrame

import torch


class ArcFaceEvaluator:
    def __init__(self, initial_padding=0.5):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.initial_padding = initial_padding  # automatically adds black padding to images before face detection to work better with big faces

    def detect_face(self, img):
        if self.initial_padding != 1.:
            H, W = img.shape[:2]
            pad_y, pad_x = int(self.initial_padding * H), int(self.initial_padding * W)
            img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))

        detected_face = None
        for j in range(4):
            detection_results = self.app.get(img[:, :, ::-1], max_num=1)
            if len(detection_results) > 0:
                detected_face = detection_results[0]
                break
            else:
                H, W = img.shape[:2]
                img = np.pad(img, ((H // 2, H // 2), (W // 2, W // 2), (0, 0)))
        return detected_face

    def __call__(self, pred_img, gt_img):
        """
        assumes pred_img, gt_img to be np.arrays of shape H x W x 3 normalized from 0 ... 255 in RGB format
        """
        pred_face = self.detect_face(pred_img)
        gt_face = self.detect_face(gt_img)

        if gt_face is None:
            return np.nan
        elif pred_face is None:
            return 0.0
        else:
            pred_emb = pred_face.normed_embedding
            gt_emb = gt_face.normed_embedding
            similarity = np.sum(pred_emb * gt_emb)

            return similarity


@torch.no_grad()
def evaluate_folder(prediction_folder, nsamples=-1, use_mask=True, eval_resolution=None, gt_folder=None):
    accelerator = Accelerator()
    arcface_evaluator = ArcFaceEvaluator()
    mse_evaluator = torchmetrics.MeanSquaredError().to(accelerator.device)
    lpips_evaluator = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(normalize=True).to(accelerator.device)
    ssim_evaluator = torchmetrics.image.ssim.StructuralSimilarityIndexMeasure(data_range=(0., 1.0)).to(accelerator.device)
    gt_folder = Path(gt_folder if gt_folder is not None else prediction_folder)

    pred_imgs = [Path(f) for f in sorted(glob.glob(prediction_folder + "/*_pred.png"))]
    if nsamples > 0:
        pred_imgs = pred_imgs[:nsamples]
    gt_imgs = [gt_folder / (f.name.split("_")[0] + "_gt.png") for f in pred_imgs]
    mask_imgs = [gt_folder / (f.name.split("_")[0] + "_mask.png") for f in pred_imgs]

    detail_results = dict(
        image_alignment=[],
        rgb_mse=[],
        lpips=[],
        ssim=[],
    )

    for i, (pred_path, gt_path, mask_path) in tqdm(enumerate(zip(pred_imgs, gt_imgs, mask_imgs)), desc="Evaluation", total=len(pred_imgs)):
        gt_img = Image.open(str(gt_path)).convert("RGB")
        pred_img = Image.open(str(pred_path)).convert("RGB")
        mask_img = Image.open(str(mask_path))

        if eval_resolution is not None:
            gt_img = gt_img.resize((eval_resolution, eval_resolution))
            pred_img = pred_img.resize((eval_resolution, eval_resolution))
            mask_img = mask_img.resize((eval_resolution, eval_resolution))

        pred_img_np = np.array(pred_img)
        gt_img_np = np.array(gt_img)
        pred_img_pt = torchvision.transforms.functional.to_tensor(pred_img).to(accelerator.device)  # 3 x H x W
        gt_img_pt = torchvision.transforms.functional.to_tensor(gt_img).to(accelerator.device)
        mask_pt = torchvision.transforms.functional.to_tensor(mask_img).to(accelerator.device)

        identity_similarity = arcface_evaluator(pred_img_np, gt_img_np)
        # pose = pose_evaluator(pred_img_np, gt_img_np)
        if use_mask:
            pred_img_pt = pred_img_pt * mask_pt
            gt_img_pt = gt_img_pt * mask_pt
        rgb_mse = mse_evaluator(pred_img_pt, gt_img_pt).cpu().item()
        lpips = lpips_evaluator(pred_img_pt[None], gt_img_pt[None]).cpu().item()
        ssim = ssim_evaluator(pred_img_pt[None], gt_img_pt[None]).cpu().item()
        # fid_evaluator.update(pred_img_pt[None], real=False)
        # fid_evaluator.update(gt_img_pt[None], real=True)

        detail_results["image_alignment"].append(identity_similarity)
        detail_results["rgb_mse"].append(rgb_mse)
        detail_results["lpips"].append(lpips)
        detail_results["ssim"].append(ssim)
        # detail_results["pose"].append(pose)

    # write detail results
    write_dict = dict(image_name=[p.name for p in pred_imgs], **detail_results)
    df_detail = DataFrame(data=write_dict)
    outfpath = os.path.join(prediction_folder, "score_detail.txt")
    df_detail.to_csv(outfpath, header=True)

    # write averaged results
    results = dict([(k, [nan_safe_mean(v), nan_safe_std(v) / np.sqrt(no_nan_len(v))]) for k, v in detail_results.items()])
    outfpath = os.path.join(prediction_folder, "score.txt")
    with open(outfpath, "w") as f:
        for k in results:
            f.write(f"{k}: {results[k][0]} +- {results[k][1]}\n")
    os.system(f"cat {outfpath}")

    del arcface_evaluator
    del mse_evaluator
    del lpips_evaluator
    del ssim_evaluator
    torch.cuda.empty_cache()

    return results


def no_nan_len(l):
    return len([x for x in l if not np.isnan(x)])


def nan_safe_mean(l):
    return np.mean([x for x in l if not np.isnan(x)])


def nan_safe_std(l):
    return np.std([x for x in l if not np.isnan(x)])


if __name__ == "__main__":
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('10.34.31.195', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    # export LD_LIBRARY_PATH=/is/software/nvidia/cudnn-8.7.0-cu11.x/lib:$LD_LIBRARY_PATH
    evaluate_folder(prediction_folder="", nsamples=-1, use_mask=True, ref_is_gt=True)
