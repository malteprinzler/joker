"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""

import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
import nvdiffrast.torch as dr
from scipy.io import loadmat
from torch import nn


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0],
                     [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


def ndc_projection_from_bbx(n, f, focal_len, l, t, r, b):
    if not (isinstance(focal_len, list) or isinstance(focal_len, tuple)):
        focal_len = [focal_len, focal_len]
    l = l / focal_len[0] * n
    t = t / focal_len[1] * n
    r = r / focal_len[0] * n
    b = b / focal_len[1] * n

    return np.array(
        [[2 * n / (r - l), 0, (r + l) / (r - l), 0],
         [0, 2 * n / (t - b), 1 * (t + b) / (t - b), 0],
         [0, 0, -1 * (f + n) / (f - n), -2 * f * n / (f - n)],
         [0, 0, -1, 0]
         ]
    ).astype(np.float32)


class MeshRenderer(nn.Module):
    def __init__(self,
                 rasterize_fov,
                 znear=0.1,
                 zfar=10,
                 rasterize_size=224,
                 use_opengl=True):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
            torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.use_opengl = use_opengl
        self.ctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = self.rasterize_size
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.ctx is None:
            if self.use_opengl:
                self.ctx = dr.RasterizeGLContext(device=device)
                ctx_str = "opengl"
            else:
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "cuda"
            print("create %s ctx on device cuda:%d" % (ctx_str, device.index))

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        if isinstance(rsize, int):
            rsize = [rsize, rsize]
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=rsize, ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image
