import os
from pathlib import Path
from pytorch3d.io import load_obj
import torch
import json
from omegaconf import OmegaConf
import inferno.utils.DecaUtils as util
from inferno.models.FaceReconstruction.FaceRecBase import shape_model_from_cfg, renderer_from_cfg
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

mesh_path = Path("mesh.obj")
camera_root = Path("")
mesh_out_path = camera_root.parent / "flame_mesh_pl" / "flame_mesh.obj"

device = torch.device("cuda")
mesh = load_obj(mesh_path)
verts = mesh[0].to(device)
faces = mesh[1].verts_idx.to(device)

# loading flame shape model + renderer
cfg = OmegaConf.load("../../assets/FaceReconstruction/models/EMICA-CVT_flame2020_notexture/cfg.yaml")
cfg.model.renderer.type = "PerspectiveDecaRenderer"
renderer = renderer_from_cfg(cfg)
renderer.to(device)

cam_paths = [p for p in sorted(camera_root.iterdir())]

for p in cam_paths:
    with open(p, "r") as f:
        cam_params = json.load(f)
    intrinsics = torch.tensor(cam_params["intrinsics"], device=device)[None]
    extrinsics = torch.tensor(cam_params["extrinsics"], device=device)[None]
    lightcode = torch.zeros((1, 9, 3), device=device, dtype=torch.float)
    image_original = torch.zeros((1, 3, 512, 512), device=device, dtype=torch.float)
    tform = torch.eye(3).to(image_original)[None]
    sample = dict(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        lightcode=lightcode,
        image_original=image_original,
        image=image_original,
        tform=tform,
        verts=verts[None],
    )
    sample = renderer(sample)
    normals = util.vertex_normals(verts[None], faces[None])
    normal_images = renderer.render.render_normal(sample["trans_verts"], normals)

    # rotating normals into camera space
    extrinsics = sample["extrinsics"]
    H_orig = sample["H_orig"][0]
    normals_cam = (extrinsics[:, :3, :3] @ normal_images.reshape(1, 3, -1)).reshape(1, 3, H_orig, H_orig)
    normals_cam[:, 1:] *= -1
    normal_mask = torch.any(normal_images != 0, dim=1, keepdim=True).float()
    normals_cam = normals_cam * normal_mask
    normal_images = normals_cam

    normal_image_np = np.clip(np.round(normal_images[0].detach().cpu().permute(1, 2, 0).numpy() * 127.5 + 127.5), a_min=0, a_max=255).astype(np.uint8)
    normal_image_np[torch.all(normal_images[0] == 0, dim=0).cpu().numpy()] = 0

    normal_out_path = p.parents[1] / "flame_normals_pl-512" / p.name.replace(".json", ".jpg")
    normal_out_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(normal_out_path, normal_image_np)

mesh_out_path.parent.mkdir(exist_ok=True, parents=True)
os.system(f"cp {mesh_path} {mesh_out_path}")
