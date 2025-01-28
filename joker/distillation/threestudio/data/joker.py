import copy
import math
import random
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
from itertools import product

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from torch.utils.data import DataLoader

from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
)
from threestudio.utils.typing import *
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
import json
from threestudio.data.random_multiview import RandomMultiviewCameraIterableDataset
from packaging import version as pver


def get_projection_matrix_from_opencv(intrinsics_cv, near, far, H, W):
    """
    taken from https://amytabb.com/tips/tutorials/2019/06/28/OpenCV-to-OpenGL-tutorial-essentials/
    intrinsics_cv: N x 3 x 3

    """
    N = len(intrinsics_cv)
    fx = intrinsics_cv[:, 0, 0]
    fy = intrinsics_cv[:, 1, 1]
    cx = intrinsics_cv[:, 0, -1]
    cy = intrinsics_cv[:, 1, -1]

    A = -(near + far)
    B = near * far

    K_gl = torch.zeros((N, 4, 4), device=intrinsics_cv.device, dtype=intrinsics_cv.dtype)
    K_gl[:, 0, 0] = -fx
    K_gl[:, 0, 2] = -(W - cx)
    K_gl[:, 1, 1] = -fy
    K_gl[:, 1, 2] = -(H - cy)
    K_gl[:, 2, 2] = A
    K_gl[:, 2, 3] = B
    K_gl[:, 3, 2] = 1

    # K_gl = torch.tensor(
    #     [
    #         [-fx, 0, -(W - cx), 0],
    #         [0, -fy, -(H - cy), 0],
    #         [0, 0, A, B],
    #         [0, 0, 1, 0]
    #     ],
    # )

    NDC_gl = torch.tensor(
        [
            [-2 / W, 0, 0, 1],
            [0, 2 / H, 0, -1],
            [0, 0, -2 * (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], device=intrinsics_cv.device, dtype=intrinsics_cv.dtype
    ).repeat(N, 1, 1)

    K_out = NDC_gl @ K_gl
    return K_out


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, convention="opencv"):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    assert convention in ["opencv", "opengl"]
    # opencv convention: +x = right +y = down, +z = look-at
    # opengl convention +x = right, +y = up, -z = look-at

    device = poses.device
    B = poses.shape[0]

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    if convention == "opengl":
        #  flipping of y axis
        j = H - j

    results = {}

    if N > 0:
        N = min(N, H * W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device) if patch_size < H else torch.zeros(size=[num_patch], device=device, dtype=torch.int)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device) if patch_size < W else torch.zeros(size=[num_patch], device=device, dtype=torch.int)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    results['inds'] = inds

    cx = intrinsics[:, 0, -1].unsqueeze(-1)
    cy = intrinsics[:, 1, -1].unsqueeze(-1)
    fx = intrinsics[:, 0, 0].unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    if convention == "opengl":
        # flipping of z axis
        zs *= -1

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # # visualize ray directions
    # direction_img = directions.cpu().reshape(B, H, W, 3)[0] * .5 + .5
    # import matplotlib.pyplot as plt
    # plt.imshow(direction_img)
    # plt.show()

    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def invert_pose(pose):
    # pose: N x 4 x 4
    inv_pose = torch.zeros_like(pose)
    inv_pose[:, -1, -1] = 1
    inv_pose[:, :3, :3] = pose[:, :3, :3].transpose(-1, -2)
    inv_pose[:, :3, -1:] = -pose[:, :3, :3].transpose(-1, -2) @ pose[:, :3, -1:]
    return inv_pose


def resize_intrinsics(K, l_in, l_out):
    """
    K: intrinsics tensor of shape N x 3 x 3
    """
    f = l_out / l_in
    K_new = copy.deepcopy(K)
    K_new[:, :2] *= f
    return K_new


def load_caption(p):
    with open(p, "r") as f:
        caption = f.readline().strip().lower()
    return caption


@dataclass
class JokerDataModuleConfig(RandomCameraDataModuleConfig):
    root: str = ""
    val_root: Optional[str] = None
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    target_grid_v: int = 0
    target_grid_h: int = 0
    patch_size: int = 64
    caption_dir: str = "blip_captions"


class JokerParentDataset:
    '''
    camera convention:
    - pixel coordinates: using opengl convention: +x = right, +y = up, -z = look-at, (0,0) lies at bottom left corner of bottom left pixel
    '''
    device = None
    cfg = None
    samples = list()
    evaluation_samples = None
    target_cam_idcs = None

    def load_samples(self):
        samples = list()
        cam_paths = sorted((Path(self.root) / "cam-512").iterdir())
        for p in cam_paths:
            # loading camera parameters
            with open(p, "r") as f:
                cam_params = json.load(f)
            extrinsics = torch.tensor(cam_params["extrinsics"], device=self.device)
            joker_2_nerf = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]], device=self.device)
            extrinsics_nerf = joker_2_nerf @ extrinsics
            intrinsics = torch.tensor(cam_params["intrinsics"], device=self.device)
            intrinsics[1, 2] = 512 - intrinsics[1, 2]  # have to also flip y axis of intrinsics

            # loading images & captions
            img_path = p.parents[1] / "images-512" / p.name.replace(".json", ".jpg")
            mask_path = p.parents[1] / "mask-512" / p.name.replace(".json", ".png")
            ctrl_path = p.parents[1] / "bfm_normals-512" / p.name.replace(".json", ".jpg")
            caption_path = list((p.parents[1] / self.cfg.caption_dir).iterdir())[0]

            ctrl = read_image(str(ctrl_path), ImageReadMode.RGB).float().to(self.device).permute(1, 2, 0) / 255  # [H, W, 3] (0...1)
            if img_path.exists():
                img = read_image(str(img_path), ImageReadMode.RGB).float().to(self.device).permute(1, 2, 0) / 255  # [H, W, 3] (0...1)
            else:
                img = torch.ones_like(ctrl)
            mask = torch.ones_like(img[..., :1])  # read_image(str(mask_path), ImageReadMode.GRAY).float().to(self.device).permute(1, 2, 0) / 255  # [H, W, 1] (0...1)
            caption = load_caption(caption_path)

            samples.append(dict(
                extrinsics=extrinsics,
                extrinsics_nerf=extrinsics_nerf,
                intrinsics=intrinsics,
                ctrl=ctrl,
                img=img,
                mask=mask,
                caption=caption
            ))

        # loading ref img (same for all samples)
        ref_path = p.parents[1] / "ref_img" / "ref_img.jpg"
        if ref_path.exists():
            ref = read_image(str(ref_path), ImageReadMode.RGB).float().to(self.device).permute(1, 2, 0).unsqueeze(0) / 255  # [Nref, H, W, 3] (0...1)
        else:
            warnings.warn("NO REFERENCE IMAGE FOUND")
            ref = torch.zeros_like(ctrl).unsqueeze(0)  # [Nref, H, W, 3] (0...1)
        for sample in samples:
            sample["ref"] = ref
        return samples

    def load_evaluation_samples(self):
        p = Path(self.root) / "target_cam-512" / "target_cam.json"

        # loading camera parameters
        with open(p, "r") as f:
            cam_params = json.load(f)
        extrinsics = torch.tensor(cam_params["extrinsics"], device=self.device)
        joker_2_nerf = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]], device=self.device)
        extrinsics_nerf = joker_2_nerf @ extrinsics
        intrinsics = torch.tensor(cam_params["intrinsics"], device=self.device)
        intrinsics[1, 2] = 512 - intrinsics[1, 2]  # have to also flip y axis of intrinsics

        # loading images & captions
        img_path = p.parents[1] / "target_img" / "target_img.jpg"
        mask_path = p.parents[1] / "target_mask" / "target_mask.png"
        ref_path = p.parents[1] / "ref_img" / "ref_img.jpg"
        img = read_image(str(img_path), ImageReadMode.RGB).float().to(self.device).permute(1, 2, 0) / 255  # [H, W, 3] (0...1)
        ctrl = torch.zeros_like(img)  # [H, W, 3] (0...1)
        mask = read_image(str(mask_path), ImageReadMode.GRAY).float().to(self.device).permute(1, 2, 0) / 255  # [H, W, 1] (0...1)
        caption = ""
        ref = read_image(str(ref_path), ImageReadMode.RGB).float().to(self.device).permute(1, 2, 0).unsqueeze(0) / 255  # [Nref, H, W, 3] (0...1)

        samples = [dict(
            extrinsics=extrinsics,
            extrinsics_nerf=extrinsics_nerf,
            intrinsics=intrinsics,
            ctrl=ctrl,
            img=img,
            mask=mask,
            caption=caption,
            ref=ref
        )]

        return samples

    # def get_random_ray_batch(self, nrays_per_sample):
    #     batch_size = len(self.samples)
    #
    #     # rescale intrinsics and imgs
    #     # Importance note: the returned rays_d MUST be normalized!
    #     ray_dict = get_rays(self.all_c2w, self.all_intrinsics, 512, 512, N=nrays_per_sample, error_map=None, patch_size=-1, convention="opengl")
    #     rays_o, rays_d, rays_ind = ray_dict["rays_o"], ray_dict["rays_d"], ray_dict["inds"]
    #     rays_u = rays_ind % 512   # ray x coordinate starting from left
    #     rays_v = rays_ind // 512   # ray y coordinate starting from top
    #
    #
    #     rays_o = rays_o.view(batch_size, nrays_per_sample, 1, 3)
    #     rays_d = rays_d.view(batch_size, nrays_per_sample, 1, 3)
    #     rays_uv = torch.cat([rays_u.view(batch_size, nrays_per_sample, 1, 1), rays_v.view(batch_size, nrays_per_sample, 1, 1)], dim=-1)
    #
    #     proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_opencv(
    #         self.all_intrinsics, 0.1, 1000.0, H=512, W=512
    #     )  # FIXME: hard-coded near and far
    #
    #     mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.all_c2w, proj_mtx)
    #     camera_positions = self.all_c2w[:, :3, -1]
    #
    #     return {
    #         "rays_o": rays_o,
    #         "rays_d": rays_d,
    #         "rays_uv": rays_uv,
    #         "mvp_mtx": mvp_mtx,
    #         "camera_positions": camera_positions,
    #         "c2w": self.all_c2w,
    #         "light_positions": camera_positions,
    #         "height": 512,
    #         "width": 512,
    #         "sample_idcs": list(range(len(self.samples))),
    #     }

    def get_random_patch_batch(self, sample_idcs, patch_size, height=None, width=None):
        batch_size = len(sample_idcs)
        height = height if height is not None else self.height
        width = width if width is not None else self.width
        patch_size = min(patch_size, height, width)  # patch size cant be bigger than height and width

        c2w = torch.stack([self.all_c2w[i] for i in sample_idcs])
        intrinsics = torch.stack([self.all_intrinsics[i] for i in sample_idcs])
        intrinsics = resize_intrinsics(intrinsics, 512, height)

        # rescale intrinsics and imgs
        # Importance note: the returned rays_d MUST be normalized!
        ray_dict = get_rays(c2w, intrinsics, height, width, N=patch_size ** 2, error_map=None, patch_size=patch_size, convention="opengl")
        rays_o, rays_d, rays_ind = ray_dict["rays_o"], ray_dict["rays_d"], ray_dict["inds"]
        rays_u = rays_ind % width  # ray x coordinate starting from left
        rays_v = rays_ind // width  # ray y coordinate starting from top

        rays_o = rays_o.view(batch_size, patch_size, patch_size, 3)
        rays_d = rays_d.view(batch_size, patch_size, patch_size, 3)
        rays_uv = torch.cat([rays_u.view(batch_size, patch_size, patch_size, 1), rays_v.view(batch_size, patch_size, patch_size, 1)], dim=-1)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_opencv(
            intrinsics, 0.1, 1000.0, H=height, W=width
        )  # FIXME: hard-coded near and far

        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        camera_positions = c2w[:, :3, -1]

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rays_uv": rays_uv,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": camera_positions,
            "height": height,
            "width": width,
            "sample_idcs": sample_idcs,
        }

    def get_data_batch(self, sample_idcs, n_view=None, height=None, width=None, evaluation_samples=False):
        batch_size = len(sample_idcs)
        n_view = n_view if n_view is not None else self.cfg.n_view
        height = height if height is not None else self.height
        width = width if width is not None else self.width

        samples = self.samples if not evaluation_samples else self.evaluation_samples

        extrinsics = torch.stack([samples[i]["extrinsics_nerf"] for i in sample_idcs])
        intrinsics = torch.stack([samples[i]["intrinsics"] for i in sample_idcs])
        imgs = torch.stack([samples[i]["img"] for i in sample_idcs])
        ctrl = torch.stack([samples[i]["ctrl"] for i in sample_idcs])
        ref = torch.stack([samples[i]["ref"] for i in sample_idcs])
        masks = torch.stack([samples[i]["mask"] for i in sample_idcs])

        c2w = invert_pose(extrinsics)
        camera_positions = c2w[:, :3, -1]

        cam_azs = torch.arctan(camera_positions[:, 0] / camera_positions[:, 2]) * 180 / torch.pi
        cam_els = torch.arctan(camera_positions[:, 1] / camera_positions[:, 2]) * 180 / torch.pi

        # rescale intrinsics and imgs
        intrinsics = resize_intrinsics(intrinsics, 512, height)
        imgs = torchvision.transforms.functional.resize(imgs.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)  # N x H x W x 3

        # Importance note: the returned rays_d MUST be normalized!
        ray_dict = get_rays(c2w, intrinsics, height, width, N=-1, error_map=None, patch_size=-1, convention="opengl")
        rays_o, rays_d = ray_dict["rays_o"], ray_dict["rays_d"]
        rays_o = rays_o.view(batch_size, height, width, 3)
        rays_d = rays_d.view(batch_size, height, width, 3)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix_from_opencv(
            intrinsics, 0.1, 1000.0, H=height, W=width
        )  # FIXME: hard-coded near and far

        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # # visualizing cameras and rays:
        # import matplotlib.pyplot as plt
        # all_extrinsics = invert_pose(c2w).cpu()
        #
        # rots_cam2world = all_extrinsics[:, :3, :3].permute(0, 2, 1)  # N, 3, 3
        # cam_centers = (-rots_cam2world @ all_extrinsics[:, :3, -1:])  # N, 3, 1
        # cam_ids = list(range(len(all_extrinsics)))
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # s = .1
        #
        # # Drawing Cameras
        # for i, color in enumerate(["red", "green", "blue"]):
        #     ax.quiver(cam_centers[:, 0, 0],
        #               cam_centers[:, 1, 0],
        #               cam_centers[:, 2, 0],
        #               s * rots_cam2world[:, 0, i],
        #               s * rots_cam2world[:, 1, i],
        #               s * rots_cam2world[:, 2, i],
        #               edgecolor=color)
        #
        #     # Annotating Cameras
        # for i, id in enumerate(cam_ids):
        #     ax.text(cam_centers[i, 0, 0],
        #             cam_centers[i, 1, 0],
        #             cam_centers[i, 2, 0],
        #             str(id))
        #
        # # drawing rays:
        # rays_o_flattened = rays_o.reshape(-1, 3).cpu()
        # rays_d_flattened = rays_d.reshape(-1, 3).cpu()
        # ray_idcs = torch.randperm(len(rays_o_flattened))[:200]
        # ax.quiver(
        #     rays_o_flattened[ray_idcs, 0],
        #     rays_o_flattened[ray_idcs, 1],
        #     rays_o_flattened[ray_idcs, 2],
        #     s * rays_d_flattened[ray_idcs, 0],
        #     s * rays_d_flattened[ray_idcs, 1],
        #     s * rays_d_flattened[ray_idcs, 2],
        #     edgecolor="pink",
        # )
        #
        # # setting correct and equally scaled view frustum
        # bbx = cam_centers[..., 0].min(dim=0).values, cam_centers[..., 0].max(dim=0).values
        # bbx_side_length = torch.max(bbx[1] - bbx[0])
        # bbx_center = 0.5 * (bbx[0] + bbx[1])
        # co_min = bbx_center - bbx_side_length / 2
        # co_max = bbx_center + bbx_side_length / 2
        # ax.set_xlim(co_min[0], co_max[0])
        # ax.set_ylim(co_min[1], co_max[1])
        # ax.set_zlim(co_min[2], co_max[2])
        #
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()
        #
        # ## end

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "height": height,
            "width": width,
            "imgs": imgs,
            "ctrl": ctrl,
            "ref": ref,
            "masks": masks,
            "sample_idcs": sample_idcs,
            "cam_azs": cam_azs,
            "cam_els": cam_els
        }

    def get_caption(self):
        caption_path = list((Path(self.root) / self.cfg.caption_dir).iterdir())[0]
        with open(caption_path, "r") as f:
            caption = f.readline().strip().lower()
        return caption


class JokerTrainDataset(RandomMultiviewCameraIterableDataset, JokerParentDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = Path(self.cfg.root)
        self.device = torch.device("cuda:0")
        self.samples = self.load_samples()
        self.target_cam_idcs = list(range(len(self.samples)))
        self.all_extrinsics = torch.stack([sample["extrinsics_nerf"] for sample in self.samples])
        self.all_intrinsics = torch.stack([sample["intrinsics"] for sample in self.samples])
        self.all_c2w = invert_pose(self.all_extrinsics)

    def collate(self, batch) -> Dict[str, Any]:
        assert self.batch_size % self.cfg.n_view == 0, f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        sample_idcs = random.sample(range(len(self.samples)), self.batch_size)
        return self.get_random_patch_batch(sample_idcs, self.cfg.patch_size)


class JokerValDataset(RandomCameraDataset, JokerParentDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, split="val", **kwargs)
        self.root = Path(self.cfg.root)
        self.device = torch.device("cuda:0")
        self.samples = self.load_samples()
        self.evaluation_samples = self.load_evaluation_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_batch = self.get_data_batch([idx], n_view=1, height=self.cfg.eval_height, width=self.cfg.eval_width)
        sample = dict(index=idx)
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.squeeze(0)
            else:
                sample[k] = v
        return sample

    def get_evaluation_batch(self, idcs=[0]):
        batch = self.get_data_batch(idcs, n_view=1, height=self.cfg.eval_height, width=self.cfg.eval_width, evaluation_samples=True)
        batch["index"] = idcs
        batch["height"] = self.cfg.eval_height
        batch["width"] = self.cfg.eval_width
        return batch

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("joker-datamodule")
class JokerDataModule(pl.LightningDataModule):
    cfg: JokerDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(JokerDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = JokerTrainDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            cfg = copy.deepcopy(self.cfg)
            if cfg.val_root != "":
                cfg.root = cfg.val_root
            self.val_dataset = JokerValDataset(cfg)
        if stage in [None, "test", "predict"]:
            # cfg = copy.deepcopy(self.cfg)
            # if cfg.val_root != "":
            #     cfg.root = cfg.val_root
            self.test_dataset = JokerValDataset(self.cfg)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


if __name__ == "__main__":
    ds = JokerTrainDataset(cfg=JokerDataModuleConfig(
        root="",
        target_grid_h=20,
        target_grid_v=20,
        patch_size=64
    ))
    ds.batch_size = 64

    # ds.get_target_cam_idcs(1)
    # ds.collate(None)
    ds.get_data_batch(list(range(400)))
