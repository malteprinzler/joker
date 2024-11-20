import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import tqdm
from omegaconf import OmegaConf
# import plotly.graph_objects as go
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from pathlib import Path
from functools import partial
from collections import defaultdict
from torch import multiprocessing
import sys

sys.path = ["joker/third_party/inferno"] + sys.path
from inferno.models.DecaFLAME import FLAME

flame_config = OmegaConf.create(
    dict(
        flame_model_path="assets/inferno/FLAME/geometry/generic_model.pkl",
        n_shape=300,
        n_exp=100,
        flame_lmk_embedding_path="assets/inferno/FLAME/geometry/landmark_embedding.npy"
    )
)
flame = FLAME(flame_config)
FLAME_SHAPEDIR_NORMS = (flame.shapedirs.flatten(end_dim=1) ** 2).sum(dim=0).sqrt()
FLAME_EXPR_SHAPEDIR_NORMS = FLAME_SHAPEDIR_NORMS[300:]


def get_pose_distance(pose1, pose2):
    pose1 = R.from_rotvec(pose1.cpu().numpy())
    pose2 = R.from_rotvec(pose2.cpu().numpy())

    diff_pose = pose1 * pose2.inv()
    diff_pose = diff_pose.as_matrix()
    theta = np.arccos((np.trace(diff_pose) - 1) / 2)
    return theta


# getting pose diversity and extreme-ness
def flame_param_distance(p1, p2=None, w_global=1., w_jaw=1.):
    expr1 = torch.tensor(p1["expcode"])
    globalpose1 = torch.tensor(p1["globalpose"])
    jawpose1 = torch.tensor(p1["jawpose"])

    if p2 is None:
        expr2 = torch.zeros(100, dtype=torch.float)
        globalpose2 = torch.zeros(3, dtype=torch.float)
        jawpose2 = torch.zeros(3, dtype=torch.float)
    else:
        expr2 = torch.tensor(p2["expcode"])
        globalpose2 = torch.tensor(p2["globalpose"])
        jawpose2 = torch.tensor(p2["jawpose"])

    expr_distance = (((expr1 - expr2) * FLAME_EXPR_SHAPEDIR_NORMS / FLAME_EXPR_SHAPEDIR_NORMS.sum()) ** 2).sum().sqrt()
    globalpose_distance = get_pose_distance(globalpose1, globalpose2)
    jawpose_distance = get_pose_distance(jawpose1, jawpose2)

    total_distance = expr_distance + globalpose_distance * w_global + jawpose_distance * w_jaw
    return total_distance


def img_path_2_normal_path(p):
    frame_dir = p.parents[1]
    normal_path = frame_dir / "flame_normals-512" / (p.stem + ".jpg")
    return normal_path


def sort_sequence_by_expressiveness(seq_path: Path, w_global, w_jaw, cam_name):
    frame_dirs = [p for p in seq_path.iterdir() if p.name.startswith("frame_") and p.is_dir()]
    param_files = [p / "flame_params-512" / f"{cam_name}.json" for p in frame_dirs]
    expressivenesses = list()
    for p in param_files:
        with open(p, "r") as f:
            flame_params = json.load(f)
            expressivenesses.append(flame_param_distance(flame_params, w_global=w_global, w_jaw=w_jaw))

    sorted_idcs = np.argsort(expressivenesses)[::-1]
    fig, axes = plt.subplots(ncols=len(frame_dirs), nrows=2)
    for i, idx in enumerate(sorted_idcs):
        frame_dir = frame_dirs[idx]
        expressiveness = expressivenesses[idx]
        normal_path = frame_dir / "flame_normals-512" / f"{cam_name}.jpg"
        img_path = frame_dir / "images-512" / f"{cam_name}.jpg"
        img = Image.open(img_path)
        normal_img = Image.open(normal_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(frame_dir.name)
        axes[1, i].imshow(normal_img)
        t = axes[1, i].text(0.1, 0.1, f"{expressiveness:.2e}", transform=axes[1, i].transAxes, fontsize=10)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    [ax.axis("off") for ax in axes.flatten()]
    plt.show()


def get_subject_divex_metas(frame_dirs, camname, root, w_global=0.5, w_jaw=1.0, diversity_scale_factor=2., tongue_bonus=0.):
    flame_params = list()
    img_paths = list()
    nframes = len(frame_dirs)
    captions = list()
    for frame_dir in frame_dirs:
        img_path = frame_dir / "images-512" / f"{camname}.jpg"
        flame_param_path = frame_dir / "flame_params-512" / f"{camname}.json"
        caption_path = frame_dir / "blip_captions" / f"{camname}.txt"
        with open(flame_param_path, "r") as f:
            flame_params.append(json.load(f))
        with open(caption_path, "r") as f:
            captions.append(f.readline().strip())
        img_paths.append(img_path)

    expressivenesses = np.array([flame_param_distance(p, w_global=w_global, w_jaw=w_jaw) for p in flame_params])
    has_tongues = np.array([1. if "tongue" in c.lower() else 0. for c in captions])
    expressivenesses += has_tongues * tongue_bonus
    diversities = np.array([[flame_param_distance(p1, p2, w_global=w_global, w_jaw=w_jaw) for p1 in flame_params] for p2 in flame_params])
    diversities_argsort = np.argsort(diversities, axis=1)
    divex_sorted_idcs = list()
    divex_metas = list()
    for i in range(nframes):
        max_diversities = np.max(diversities, axis=1)
        if len(divex_sorted_idcs) > 0:
            max_diversities = np.minimum(max_diversities, np.min(diversities.T[divex_sorted_idcs].T, axis=1))  # penalizing diversity of samples that are redundant with already registered samples
        divex_scores = expressivenesses + max_diversities * diversity_scale_factor
        divex_scores[divex_sorted_idcs] = 0  # dont use selected ids again
        best_idx = np.argmax(divex_scores)
        divex_sorted_idcs.append(best_idx)
        divex_metas.append(
            dict(path=str(img_paths[best_idx].relative_to(root)),
                 divex=float(divex_scores[best_idx]),
                 expressiveness=float(expressivenesses[best_idx]),
                 max_diversity=float(max_diversities[best_idx]),
                 farest_neighbors=[str(img_paths[j].relative_to(root)) for j in diversities_argsort[best_idx][:-6:-1]],
                 diversities=[float(diversities[best_idx, j]) for j in diversities_argsort[best_idx][:-6:-1]]
                 )
        )
    return divex_metas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates DIVEX (Diversity and Expressiveness) metadata for samples")
    parser.add_argument("--root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--frame_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/VALID_FRAMES.txt")
    parser.add_argument("--cam_name", type=str, default="cam_00")
    parser.add_argument("--out_path", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/VALID_DIVEX_METAS.json")
    parser.add_argument("--tongue_bonus", type=float, default=0.)
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    # root = Path("")
    # valid_frames_file = root / "TRAIN_FRAMES.txt"
    # outpath = root / "TRAIN_DIVEX_METAS.json"
    # camname = "cam_222200037"
    # nworkers = 8
    # tongue_bonus = 1.

    frame_paths = np.loadtxt(args.frame_file, dtype=str)
    subject_frame_path_dict = defaultdict(list)
    for frame in frame_paths:
        subject = frame.split("/")[0]
        subject_frame_path_dict[subject].append(args.root / frame)
    subjects = np.sort(list(subject_frame_path_dict.keys()))
    subject_frame_path_list = [subject_frame_path_dict[s] for s in subjects]

    worker_fn = partial(get_subject_divex_metas, camname=args.cam_name, root=args.root, tongue_bonus=args.tongue_bonus)

    # for subject_frame_paths in tqdm.tqdm(subject_frame_path_list):
    #     worker_fn(subject_frame_paths)

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(args.n_workers) as p:
        divex_metas = list(tqdm.tqdm(p.imap(worker_fn, subject_frame_path_list), total=len(subject_frame_path_list)))
    divex_metas = [x for sublist in divex_metas for x in sublist]

    args.out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_path, "w") as f:
        json.dump(divex_metas, f, indent="\t")

    # # visualization of max divex samples
    # best_idcs = np.argsort([m["divex"] for m in divex_metas])[:-11:-1]
    # fig, axes = plt.subplots(ncols=10, nrows=6)
    # for i, idx in enumerate(best_idcs):
    #     img_path = root / divex_metas[idx]["path"]
    #     ref1_path = root / divex_metas[idx]["farest_neighbors"][0]
    #     ref2_path = root / divex_metas[idx]["farest_neighbors"][1]
    #
    #     normal_path = img_path_2_normal_path(img_path)
    #     ref1_normal_path = img_path_2_normal_path(ref1_path)
    #     ref2_normal_path = img_path_2_normal_path(ref2_path)
    #
    #     divex_score = divex_metas[idx]["divex"]
    #     expressiveness = divex_metas[idx]["expressiveness"]
    #     diversity_1 = divex_metas[idx]["diversities"][0]
    #     diversity_2 = divex_metas[idx]["diversities"][1]
    #
    #     axes[0, i].imshow(Image.open(img_path))
    #     axes[0, i].set_title(f"{img_path.parents[4].name}-{img_path.parents[2].name}-{img_path.parents[1].name}-{img_path.stem}", fontsize=6)
    #     t = axes[0, i].text(0.1, 0.1, f"DivEx: {divex_score:.2e}", transform=axes[0, i].transAxes, fontsize=10)
    #     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    #
    #     axes[1, i].imshow(Image.open(normal_path))
    #     t = axes[1, i].text(0.1, 0.1, f"Exp: {expressiveness:.2e}", transform=axes[1, i].transAxes, fontsize=10)
    #     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    #
    #     axes[2, i].imshow(Image.open(ref1_path))
    #     axes[2, i].set_title(f"{ref1_path.parents[4].name}-{ref1_path.parents[2].name}-{ref1_path.parents[1].name}-{ref1_path.stem}", fontsize=6)
    #     axes[3, i].imshow(Image.open(ref1_normal_path))
    #     t = axes[3, i].text(0.1, 0.1, f"Div: {diversity_1:.2e}", transform=axes[3, i].transAxes, fontsize=10)
    #     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    #
    #     axes[4, i].imshow(Image.open(ref2_path))
    #     axes[4, i].set_title(f"{ref2_path.parents[4].name}-{ref2_path.parents[2].name}-{ref2_path.parents[1].name}-{ref2_path.stem}", fontsize=6)
    #     axes[5, i].imshow(Image.open(ref2_normal_path))
    #     t = axes[5, i].text(0.1, 0.1, f"Div: {diversity_2:.2e}", transform=axes[5, i].transAxes, fontsize=10)
    #     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    #
    # [ax.axis("off") for ax in axes.flatten()]
    # plt.show()
