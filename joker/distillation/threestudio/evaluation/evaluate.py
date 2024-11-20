import copy

import PIL.Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from pathlib import Path
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


def read_reference_images(folder_path: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    return images


def compute_similarity_matrix(
    evaluator, cropped_images: List[np.ndarray], reference_images: List[np.ndarray]
) -> np.ndarray:
    similarity_matrix = np.zeros((len(cropped_images), len(reference_images)))
    for i, cropped_image in enumerate(cropped_images):
        for j, reference_image in enumerate(reference_images):
            embed1 = evaluator(cropped_image)
            embed2 = evaluator(reference_image)
            similarity_matrix[i, j] = embed1 @ embed2.T

    print(similarity_matrix)
    return similarity_matrix


def greedy_matching(scores):
    n, m = scores.shape
    assert n == m
    res = []
    for _ in range(m):
        pos = np.argmax(scores)
        i, j = pos // m, pos % m

        res.append(scores[i, j])
        scores[i, :] = -1
        scores[:, j] = -1

    return min(res)


def save_image(tensor, path):
    tensor = (tensor[0] * 0.5 + 0.5).clamp(min=0, max=1).permute(1, 2, 0) * 255.0
    tensor = tensor.cpu().numpy().astype(np.uint8)

    Image.fromarray(tensor).save(path)


def compute_average_similarity(
    idx, face_detector, face_similarity, generated_image, reference_image
) -> float:
    generated_face = face_detector(generated_image)
    reference_face = face_detector(reference_image)

    if reference_face is None:
        return np.nan
    elif generated_face is None:
        return 0.0
    else:
        reference_face = reference_face[:1]
        generated_face = generated_face[:1]

        # # visualization
        # fig, axes = plt.subplots(ncols=2)
        # axes[0].imshow(generated_face[0].permute(1, 2, 0)*.5+.5)
        # axes[1].imshow(reference_face[0].permute(1, 2, 0)*.5+.5)
        # plt.show()

        generated_face = generated_face.to(face_detector.device).reshape(1, 3, 160, 160)
        reference_face = reference_face.to(face_detector.device).reshape(1, 3, 160, 160)

        similarity = face_similarity(generated_face) @ face_similarity(reference_face).T
        return similarity.cpu().item()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--prediction_folder", type=str)
    parser.add_argument("--reference_folder", type=str)

    args = parser.parse_args(args)
    return args


def load_reference_image(reference_folder, image_id, use_2nd_image):
    path = os.path.join(reference_folder, image_id)
    image_path = sorted(glob.glob(os.path.join(path, "*.jpg")))[1 if use_2nd_image else 0]
    image = Image.open(image_path).convert("RGB")
    return image


def get_img_name(subject_id, group_name, prompt_id, prompt, instance_id, control_id=None):
    if control_id is None:
        return f"subject_{subject_id:04d}_{group_name}_prompt_{prompt_id:04d}_{prompt.replace(' ', '-')}_instance_{instance_id:04d}.jpg"
    else:
        return f"subject_{subject_id:04d}_{group_name}_prompt_{prompt_id:04d}_{prompt.replace(' ', '-')}_instance_{instance_id:04d}_control_{control_id:02d}.jpg"


def parse_img_name(fname):
    split_fname = fname.strip(".jpg").split("_")
    subject_id = int(split_fname[1])
    group = split_fname[2]
    prompt_id = int(split_fname[4])
    prompt = split_fname[5]
    instance_id = int(split_fname[7])
    control_id = int(split_fname[9]) if len(split_fname) >= 10 else None
    return dict(subject_id=subject_id, group=group, prompt_id=prompt_id, prompt=prompt, instance_id=instance_id, control_id=control_id)


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
def evaluate_imgs(preds, gts, masks=None):
    """
    preds: torch tensor N x 3 x H x W 0...1
    gts: torch tensor N x 3 x H x W 0...1
    masks: torch tensor N x 1 x H x W 0...1
    """
    device = preds.device
    # arcface_evaluator = ArcFaceEvaluator()
    mse_evaluator = torchmetrics.MeanSquaredError().to(device)
    lpips_evaluator = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    ssim_evaluator = torchmetrics.image.ssim.StructuralSimilarityIndexMeasure(data_range=(0., 1.0)).to(device)

    preds, gts = preds.clone().contiguous(), gts.clone().contiguous()
    if masks is not None:
        masks = masks.clone()

    detail_results = dict(
        # image_alignment=[],
        rgb_mse=[],
        lpips=[],
        ssim=[],
        # pose=[]
    )
    if masks is None:
        masks = [None] * len(preds)

    for i, (pred_img_pt, gt_img_pt, mask_pt) in tqdm(enumerate(zip(preds, gts, masks)), desc="Evaluation", total=len(preds)):

        pred_img_np = np.clip(np.round(pred_img_pt.permute(1, 2, 0).cpu().numpy() * 255), a_min=0, a_max=255).astype(np.uint8)  # N x H x W x 3 0...255
        gt_img_np = np.clip(np.round(gt_img_pt.permute(1, 2, 0).cpu().numpy() * 255), a_min=0, a_max=255).astype(np.uint8)  # N x H x W x 3 0...255

        # identity_similarity = arcface_evaluator(pred_img_np, gt_img_np)
        # pose = pose_evaluator(pred_img_np, gt_img_np)
        if mask_pt is not None:
            pred_img_pt = pred_img_pt * mask_pt
            gt_img_pt = gt_img_pt * mask_pt
        rgb_mse = mse_evaluator(pred_img_pt, gt_img_pt).cpu().item()
        lpips = lpips_evaluator(pred_img_pt[None], gt_img_pt[None]).cpu().item()
        ssim = ssim_evaluator(pred_img_pt[None], gt_img_pt[None]).cpu().item()
        # fid_evaluator.update(pred_img_pt[None], real=False)
        # fid_evaluator.update(gt_img_pt[None], real=True)

        # detail_results["image_alignment"].append(identity_similarity)
        detail_results["rgb_mse"].append(rgb_mse)
        detail_results["lpips"].append(lpips)
        detail_results["ssim"].append(ssim)
        # detail_results["pose"].append(pose)

    # del arcface_evaluator
    del mse_evaluator
    del lpips_evaluator
    del ssim_evaluator
    # del pose_evaluator
    torch.cuda.empty_cache()

    return detail_results


def no_nan_len(l):
    return len([x for x in l if not np.isnan(x)])


def nan_safe_mean(l):
    return np.mean([x for x in l if not np.isnan(x)])


def nan_safe_std(l):
    return np.std([x for x in l if not np.isnan(x)])


def nan_safe_append(l, v):
    if not np.isnan(v):
        l.append(v)


