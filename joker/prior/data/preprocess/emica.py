import copy
import json
import sys

sys.path = ["joker/third_party/inferno"] + sys.path
from joker.third_party.inferno.inferno_apps.FaceReconstruction.utils.load import load_model
from joker.third_party.inferno.inferno.utils.FaceDetector import FAN
from joker.third_party.inferno.inferno.datasets.ImageDatasetHelpers import bbox2point
from skimage.transform import estimate_transform
import numpy as np
import torch
import cv2
from PIL import Image


class EmicaImagePreprocessor:
    def __init__(self, device, max_detection=1, scale=1.25, crop_size=224, ):
        self.face_detector = FAN()
        self.max_detection = max_detection
        self.scale = scale
        self.resolution_inp = crop_size
        self.device = device

    def __call__(self, image: np.ndarray):
        """
        image: np.ndarray (H,W,3) 0...255
        """

        h, w, _ = image.shape
        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 1:
            print('no face detected!')
            return {"image_original": torch.tensor(image.transpose(2, 0, 1) / 255.).float()}
        else:
            old_size, center = [], []
            num_det = min(self.max_detection, len(bbox))
            for bbi in range(num_det):
                bb = bbox[0]
                left = bb[0]
                right = bb[2]
                top = bb[1]
                bottom = bb[3]
                osz, c = bbox2point(left, right, top, bottom, type=bbox_type)
                old_size += [osz]
                center += [c]

        size = []
        src_pts = []
        for i in range(len(old_size)):
            size += [int(old_size[i] * self.scale)]
            src_pts += [np.array(
                [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                 [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        dst_images = []
        tforms = []
        for i in range(len(src_pts)):
            tform = estimate_transform('similarity', src_pts[i], DST_PTS)
            tforms += [tform]
            M = tform.params.astype(np.float32)[:2]
            dst_image = cv2.warpAffine(image, M, dsize=(self.resolution_inp, self.resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)
            dst_images += [dst_image]
        dst_images = np.stack(dst_images, axis=0).astype(np.float32) / 255
        image = image.astype(np.float32) / 255
        tforms = np.stack(tforms, axis=0).astype(np.float32)

        return {'image': torch.tensor(dst_images).float().to(self.device),
                'tform': tforms,
                'image_original': torch.tensor(image.transpose(2, 0, 1)).float().to(self.device)[None],
                }


emica_model = None
emica_preprocessor = None
EMICA_MODEL_PATH = "assets/inferno/FaceReconstruction/models"
EMICA_MODEL_NAME = "EMICA-CVT_flame2020_notexture"


def init_emica():
    global emica_model, emica_preprocessor
    if emica_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emica_model, conf = load_model(EMICA_MODEL_PATH, EMICA_MODEL_NAME)
        emica_model.to(device)
        emica_model.eval()
        emica_preprocessor = EmicaImagePreprocessor(device=device)


def test(model, batch):
    batch["image"] = batch["image"].cuda()
    if len(batch["image"].shape) == 3:
        batch["image"] = batch["image"].view(1, 3, 224, 224)
    values = model(batch, training=False, validation=False)
    return values


@torch.no_grad()
def flame_from_img(img: np.ndarray):
    init_emica()
    batch = emica_preprocessor(img)
    if "image" in batch:
        try:
            vals = emica_model(batch)
            visdict = emica_model.visualize_batch(batch, None, None, in_batch_idx=None)

            for k in ["trans_verts", "verts", "image", "image_original", "predicted_image", "predicted_mask"]:
                vals.pop(k)
            for k, v in vals.items():
                v = v[0]
                if isinstance(v, torch.Tensor):
                    v = v.cpu().tolist()
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                vals[k] = v
            visdict = dict(shape_image=visdict["shape_image"][0], normal_image=visdict["normal_image"][0])

        except Exception as e:
            H, W = batch["image_original"].shape[-2:]
            vals = dict(
                shapecode=None,
                texcode=None,
                jawpose=None,
                globalpose=None,
                cam=None,
                cam_out=None,
                tform=None,
                H_orig=None,
                H_crop=None,
                lightcode=None,
                expcode=None,
            )
            visdict = dict(
                shape_image=np.zeros((H, W, 3), dtype=np.uint8),
                normal_image=np.zeros((H, W, 3), dtype=np.uint8),

            )
    else:
        H, W = batch["image_original"].shape[-2:]
        vals = dict(
            shapecode=None,
            texcode=None,
            jawpose=None,
            globalpose=None,
            cam=None,
            cam_out=None,
            tform=None,
            H_orig=None,
            H_crop=None,
            lightcode=None,
            expcode=None,
        )
        visdict = dict(
            shape_image=np.zeros((H, W, 3), dtype=np.uint8),
            normal_image=np.zeros((H, W, 3), dtype=np.uint8),
        )

    return dict(
        flame_params=vals,
        shape_image=visdict["shape_image"],
        normal_image=visdict["normal_image"],
    )


if __name__ == '__main__':
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.129', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    img = np.array(Image.open("").convert("RGB"))
    results = flame_from_img(img)
    print()
