import glob
import json
import os
import sys
import argparse
import numpy as np
import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import partial

from src.models.modnet import MODNet
from pathlib import Path
from torch import multiprocessing

root = Path("")
frame_file = root / "ALL_FRAMES.txt"
ckpt_path = "pretrained/modnet_photographic_portrait_matting.ckpt"
nworkers = 5

# create MODNet and load the pre-trained ckpt
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

if torch.cuda.is_available():
    modnet = modnet.cuda()
    weights = torch.load(ckpt_path)
else:
    weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
modnet.load_state_dict(weights)
modnet.eval()

# define image to tensor transform
im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# define hyper-parameters
ref_size = 512


def process_frame(frame, overwrite=False):
    images = [f for f in sorted((frame / "images-512").iterdir())]
    # inference images
    for im_path in images:
        matte_path = str(im_path).replace("images-512", "mask-512").replace(".jpg", ".png")
        if (not overwrite) and os.path.exists(matte_path):
            continue

        # read image
        im = Image.open(im_path)

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im).cuda()

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        assert matte_path != str(im_path)
        # print(matte_path)
        Path(matte_path).parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(matte_path)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    # check input arguments
    if not os.path.exists(root):
        print('Cannot find output path: {0}'.format(root))
        exit()
    if not os.path.exists(ckpt_path):
        print('Cannot find ckpt path: {0}'.format(ckpt_path))
        exit()

    all_frames = [root / f for f in np.loadtxt(frame_file, dtype=str)]

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    my_frames = np.array_split(all_frames, njobs)[jobid]
    print(f"Processing {len(my_frames)} out of {len(all_frames)} frames:")
    [print(f) for f in my_frames]
    print()

    worker_fn = partial(process_frame, overwrite=False)

    # with open(divex_file, "r") as f:
    #     metas = json.load(f)
    # frames = [(root / meta["path"]).parents[1] for meta in metas]

    # for frame in tqdm.tqdm(frames):
    #     process_frame(frame)

    with multiprocessing.Pool(nworkers) as p:
        r = list(tqdm.tqdm(p.imap_unordered(worker_fn, my_frames), total=len(my_frames)))
