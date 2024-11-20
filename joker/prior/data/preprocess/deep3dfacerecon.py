import json
import sys
import warnings
from pathlib import Path

from albumentations.random_utils import normal
from torch import multiprocessing
import matplotlib.pyplot as plt
import random
import numpy as np
import tqdm
from functools import partial
from PIL import Image
import torch
import copy
import os
import cv2

DEEP3DFACERECON_DIR = Path("joker/third_party/Deep3DFaceRecon_pytorch")
BFM_FOLDER = Path("assets/Deep3DFaceRecon/BFM")
DEEP3DFACERECON_CKP_DIR = Path("assets/Deep3DFaceRecon/checkpoints")
sys.path.append(str(DEEP3DFACERECON_DIR))

# getting environment variables correct for ninja extension building
os.environ["GCC"] = "gcc-9"
os.environ["CC"] = "gcc-9"
os.environ["CXX"] = "g++-9"
environ_root = Path(sys.executable).parents[1]
os.environ["CUDA_HOME"] = str(environ_root)
os.environ["CPLUS_INCLUDE_PATH"] = str(environ_root/"targets/x86_64-linux/include")  # WARNING: only tested for linux

from options.test_options import TestOptions
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from util.load_mats import load_lm3d
from models.facerecon_model import FaceReconModel
from models.bfm import ParametricFaceModel
from facenet_pytorch import MTCNN
import PIL


class Deep3DFaceReconstruction_Processor:
    def __init__(self, device, model_name="pretrained", epoch=20, ):
        bfm_folder = str(BFM_FOLDER)
        ckpts_dir = str(DEEP3DFACERECON_CKP_DIR)
        opt = TestOptions(cmd_line=f"--name={model_name} --epoch={epoch} --bfm_folder={bfm_folder} --checkpoints_dir={ckpts_dir} --use_opengl=False").parse()
        model = FaceReconModel(opt)
        model.setup(opt)
        model.device = device
        model.parallelize()
        model.eval()
        detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True,
        )
        lm3d_std = load_lm3d(opt.bfm_folder)
        face_model = ParametricFaceModel(bfm_folder=bfm_folder)

        self.opt = opt
        self.model = model
        self.detector = detector
        self.lm3d_std = lm3d_std
        self.face_model = face_model

    def get_head_coeffs(self, img, lmks=None):
        """
       assumes img to be np.ndarray of shape H x W x 3 with RGB entries normalized to 0 ... 255
       returns estimated bfm head coefficients for id, exp(pression), tex(ture), and also exact settings for rendering such as projection matrices
       """

        lmks = lmks if lmks is not None else self.detect_lmks(img)
        if lmks is None:
            return None
        else:
            h_orig, w_orig = img.shape[:2]
            img_aligned, lmks_aligned, bbx_orig = self.align_img(img, lmks, return_bbx_orig=True)

            # # visualize bbx_orig
            # import matplotlib.patches as patches
            # fig, axes = plt.subplots(ncols=2)
            # axes[0].imshow(img)
            # axes[1].imshow(img_aligned)
            # bbx = bbx_orig
            # rect = patches.Rectangle((bbx[0], bbx[1]), bbx[2] - bbx[0], bbx[3] - bbx[1], linewidth=1, edgecolor='r', facecolor='none')
            # axes[0].add_patch(rect)
            #
            # Image.fromarray(img_aligned).save("")
            # Image.fromarray(img).crop(np.round(np.array(bbx)).astype(int)).resize((224, 224)).save("")
            #
            # plt.show()

            if img_aligned is None:
                return dict(id=None, exp=None, tex=None, angle=None, gamma=None, trans=None)
            img_aligned = torch.tensor(img_aligned / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # torch tensor 1 x 3 x H x W; rgb, 0...1
            img_orig = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # torch tensor 1 x 3 x H x W; rgb, 0...1
            lmks_aligned = torch.tensor(lmks_aligned).unsqueeze(0)
            input_data = dict(imgs=img_aligned, lms=lmks_aligned,
                              bbx_orig=bbx_orig, hw_orig=[h_orig, w_orig], imgs_orig=img_orig
                              )

            self.model.set_input(input_data)  # unpack data from data loader
            self.model.test()  # run inference

            # # visualize
            # visuals = self.model.get_current_visuals()  # get image results
            # plt.imshow(visuals["output_vis"][0].permute(1, 2, 0).cpu())
            # plt.show()

            head_coeffs = copy.deepcopy(self.model.pred_coeffs_dict)
            for k, v in head_coeffs.items():
                head_coeffs[k] = v.cpu().numpy()

            head_coeffs["bbx_orig"] = np.array(bbx_orig)  # bounding box on input image (potentially also cropped), that is used as the input region for predicting bfm parameters
            head_coeffs["hw_orig"] = np.array([h_orig, w_orig])  # shape of input image (potentially also cropped) before cropping it to serve as input for predicting the bfm parameters

            return head_coeffs

    def align_img(self, img, lmk, return_bbx_orig=False, *args, **kwargs):
        H, W = img.shape[:2]
        lmk = copy.deepcopy(lmk)
        lmk[:, -1] = H - 1 - lmk[:, -1]
        img = PIL.Image.fromarray(img)
        _, img, lmk, _, bbx_orig = align_img(img, lmk, self.lm3d_std, return_bbx_orig=True, *args, **kwargs)
        if img is not None:
            img = np.array(img)
        outs = [img, lmk]
        if return_bbx_orig:
            outs.append(bbx_orig)
        return outs

    def detect_lmks(self, img):
        boxes, probs, points = self.detector.detect(img, landmarks=True)
        if points is None:
            return None
        else:
            return points[0].astype(np.float32)

    def crop_img(self, im: np.ndarray, lm: np.ndarray, blur_pad=False, return_bbx_orig=False):
        im = Image.fromarray(im, "RGB")
        _, H = im.size
        lm = copy.deepcopy(lm)
        lm[:, -1] = H - 1 - lm[:, -1]

        target_size = 1024
        rescale_factor = 300
        center_crop_size = 700
        output_size = 512

        _, im_high, _, _, bbx_orig = align_img(im, lm, self.lm3d_std, target_size=target_size, rescale_factor=rescale_factor, blur_pad=blur_pad, return_bbx_orig=True)
        if im_high is None:
            return None

        left = int(im_high.size[0] / 2 - center_crop_size / 2)
        upper = int(im_high.size[1] / 2 - center_crop_size / 2)
        right = left + center_crop_size
        lower = upper + center_crop_size
        im_cropped = im_high.crop((left, upper, right, lower))
        bbx_orig = [
            bbx_orig[0] + left / target_size * (bbx_orig[2] - bbx_orig[0]),
            bbx_orig[1] + upper / target_size * (bbx_orig[3] - bbx_orig[1]),
            bbx_orig[0] + right / target_size * (bbx_orig[2] - bbx_orig[0]),
            bbx_orig[1] + lower / target_size * (bbx_orig[3] - bbx_orig[1])
        ]

        im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
        im_cropped = np.array(im_cropped)

        if return_bbx_orig:
            return im_cropped, bbx_orig
        else:
            return im_cropped

    def headcoeff2camparams(self, headcoeff, scale_factor=1.):
        headcoeff = copy.deepcopy(headcoeff)
        angle = np.array(headcoeff['angle']).flatten()
        trans = np.array(headcoeff['trans']).flatten()

        R = self.face_model.compute_rotation(torch.from_numpy(angle)[None])[0].numpy()  # note assumes pts@R so R is actually transposed of rotation if assuming column vectors
        trans[2] += -10
        c = -np.dot(R, trans)
        c *= scale_factor
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = c

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        extrinsics = np.eye(4)
        extrinsics[:3] = np.concatenate([pose[:3, :3].T, -1 * pose[:3, :3].T @ pose[:3, -1:]], axis=-1)

        intrinsics = np.array(headcoeff["facemodel_perc_proj"]).T

        # flipping y axis of intrinsics (from origin is in bottom left to origin is in top left)
        # only have to do cy correction, since sign of focal length is correct due to extrinsic rotation
        h = headcoeff["hw_orig"][0]
        intrinsics[1, 2] = h - intrinsics[1, 2]

        cam_params = dict(intrinsics=intrinsics.tolist(),
                          extrinsics=extrinsics.tolist())
        return cam_params


device = torch.device("cuda")
deep3dface = Deep3DFaceReconstruction_Processor(device)


def predict_bfm_from_img(img_path, blur_pad=False, crop=False):
    """
    generates bfm normal map given an input image path.
    if crop: crops the image according to Joker convention before generating normal map
    if blur_pad: blur-pads the image before cropping as in FFHQ to avoid black regions after cropping

    returns:
        - img: np.ndarray (512,512,3), 0 ... 255 (cropped) image
        - normal: np.ndarray (512,512,3), 0 ... 255 normal map rendering of BFM
        - bbx_orig: bounding box of crop region on input image
        - head_coeffs: dict with bfm coefficients and other useful information
    """
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    hw_uncropped = img.shape[:2]
    lmks_uncropped = deep3dface.detect_lmks(img)

    if lmks_uncropped is None:
        print(f"WARNING: couldnt detect face for {img_path}! Doing center-crop now.")
        head_coeffs = dict(id=None, exp=None, tex=None, angle=None, gamma=None, trans=None)

        # center cropping image
        H, W, _ = img.shape
        L = min(H, W)
        x0 = W // 2 - L // 2
        y0 = H // 2 - L // 2
        quad = np.array([[x0, y0], [x0, y0 + L], [x0 + L, y0 + L], [x0 + L, y0]], dtype=np.float32)
        img = np.array(Image.fromarray(img, mode="RGB").transform((512, 512), PIL.Image.QUAD, quad.flatten(), PIL.Image.BILINEAR))
        normal = np.zeros_like(img)
        bbx = [x0, y0, x0 + L, y0 + L]

    else:
        if crop:
            img, bbx = deep3dface.crop_img(img, lmks_uncropped, blur_pad=blur_pad, return_bbx_orig=True)
            lmks = (lmks_uncropped - np.array([[bbx[0], bbx[1]]])) * np.array([[img.shape[1] / (bbx[2] - bbx[0]), img.shape[0] / (bbx[3] - bbx[1])]])
        else:
            bbx = [0, 0, hw_uncropped[1], hw_uncropped[0]]
            lmks = lmks_uncropped
        head_coeffs = deep3dface.get_head_coeffs(img, lmks)
        cam_params = deep3dface.headcoeff2camparams(head_coeffs)
        for k in ["id", "exp", "tex", "angle", "gamma", "trans"]:
            head_coeffs[k] = head_coeffs[k][0]
        head_coeffs["bbx_crop"] = bbx
        head_coeffs["lmks_cropped"] = lmks
        head_coeffs["lmks_uncropped"] = lmks_uncropped
        head_coeffs["hw_uncropped"] = hw_uncropped
        head_coeffs["hw_cropped"] = img.shape[:2]  # should be identical to hw_orig which is added to head_coeffs dict by bfm
        head_coeffs.update(cam_params)
        normal = (((deep3dface.model.pred_normal * .5 + .5) * deep3dface.model.pred_mask) * 255).round().clip(0, 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        for k, v in head_coeffs.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                head_coeffs[k] = v.tolist()

    return dict(head_coeffs=head_coeffs,
                img=img,
                normal=normal)


if __name__ == "__main__":
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.149', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # nworkers = 8
    # in_root = Path("")
    # out_root = Path("")
    # frame_file = in_root / "ALL_FRAMES.txt"
    # blur_pad = True
    #
    # njobs = int(os.getenv("CONDOR_WorldSize", 1))
    # jobid = int(os.getenv("CONDOR_Process", 0))
    # all_frames = np.loadtxt(frame_file, dtype=str)
    # all_frames = [in_root / s for s in all_frames]
    # my_frames = np.array_split(all_frames, njobs)[jobid]
    # print(f"Processing {len(my_frames)} out of {len(all_frames)} frames:")
    # [print(f) for f in my_frames]
    # print()
    #
    # worker_fn = partial(process_frame, out_root=out_root, blur_pad=blur_pad)
    #
    # for folder in tqdm.tqdm(my_frames, "Frames", leave=False):
    #     worker_fn(folder)
    #
    # # multiprocessing.set_start_method("spawn")
    # # with multiprocessing.Pool(nworkers) as p:
    # #     list(tqdm.tqdm(p.imap_unordered(worker_fn, my_frames), total=len(my_frames), desc="Frames"))

    process_results = process_img("",
                                  blur_pad=True,
                                  crop=True)
    print()
