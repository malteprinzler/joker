"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
from pathlib import Path
import glob
from glob import glob
import cv2
import numpy as np
import scipy
import torch
from skimage.io import imread
from skimage.transform import rescale, estimate_transform
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# from inferno.datasets.FaceVideoDataModule import add_pretrained_deca_to_path
from inferno.datasets.ImageDatasetHelpers import bbox2point
from inferno.utils.FaceDetector import FAN

from pytorch_lightning import LightningDataModule

import os


class TestDM(LightningDataModule):

    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        super().__init__()
        self.testpath = testpath
        self.crop = iscrop
        self.crop_size = crop_size
        self.scale = scale
        self.scaling_factor = scaling_factor
        self.face_detector = face_detector

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        self.dataset = TestData(self.testpath, iscrop=self.crop, crop_size=self.crop_size,
                                scale=self.scale, face_detector=self.face_detector,
                                scaling_factor=self.scaling_factor, max_detection=None)

    def test_dataloader(self):
        # create a data loader for self.dataset 
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        return dataloader


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None, frame_file=None):
        self.max_detection = max_detection
        self.frame_file = frame_file
        self.imagepath_list = self.get_imagepath_list(testpath)
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        # add_pretrained_deca_to_path()
        # from decalib.datasets import detectors
        if face_detector == 'fan':
            self.face_detector = FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    @staticmethod
    def get_imagepath_list(testpath):
        if isinstance(testpath, list):
            imagepath_list = testpath
        elif os.path.isdir(testpath):
            imagepath_list = glob(testpath + '/**/*.jpg', recursive=True) + glob(testpath + '/**/*.png', recursive=True) + glob(testpath + '/**/*.bmp', recursive=True)
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        return imagepath_list

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = str(self.imagepath_list[index])
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1)) * 255.

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            else:
                # bbox, bbox_type, landmarks = self.face_detector.run(image)
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 1:
                    print('no face detected!')
                    return {"image_original": torch.tensor(image.transpose(2, 0, 1) / 255.).float(), 'image_name': [imagename + f"{0:02d}"], 'image_path': [imagepath]}
                else:
                    if self.max_detection is None:
                        bbox = bbox[0]
                        left = bbox[0]
                        right = bbox[2]
                        top = bbox[1]
                        bottom = bbox[3]
                        old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
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

            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                         [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                     [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        if not isinstance(src_pts, list):
            src_pts = [src_pts]

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        dst_images = []
        tforms = []
        for i in range(len(src_pts)):
            tform = estimate_transform('similarity', src_pts[i], DST_PTS)
            tforms += [tform]
            M = tform.params.astype(np.float32)[:2]
            dst_image = cv2.warpAffine(image, M, dsize=(self.resolution_inp, self.resolution_inp))
            # dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp), order=3)
            dst_image = dst_image.transpose(2, 0, 1)
            dst_images += [dst_image]
        dst_images = np.stack(dst_images, axis=0).astype(np.float32) / 255
        image = image.astype(np.float32) / 255
        tforms = np.stack(tforms, axis=0).astype(np.float32)

        imagenames = [imagename + f"{j:02d}" for j in range(dst_images.shape[0])]
        imagepaths = [imagepath] * dst_images.shape[0]
        return {'image': torch.tensor(dst_images).float(),
                'image_name': imagenames,
                'image_path': imagepaths,
                'tform': tforms,
                'image_original': torch.tensor(image.transpose(2, 0, 1)).float()[None],
                }


class CelebAHQData(TestData):
    @staticmethod
    def get_imagepath_list(testpath):
        imagepath_list = glob(testpath + '/**/*.jpg', recursive=True)
        # imgpaths_file = os.path.join("img_paths", testpath.name + ".txt")
        # if os.path.exists(imgpaths_file):
        #     with open(imgpaths_file, "r") as f:
        #         imagepath_list = [l.strip() for l in f.readlines()]
        # else:
        #     imagepath_list = glob(testpath + '/**/*.jpg', recursive=True)
        #     with open(imgpaths_file, "w") as f:
        #         f.writelines(imagepath_list)
        return imagepath_list


class NerSembleData(TestData):

    def get_imagepath_list(self, testpath):
        # imagepath_listfile = Path(testpath).name + "_imagepath_list.txt"
        # if os.path.exists(imagepath_listfile):
        #     imagepath_list = np.loadtxt(imagepath_listfile, dtype=str)
        # else:
        #     imagepath_list = glob(testpath + '/**/images-512/*.jpg', recursive=True)
        #     with open(imagepath_listfile, "w") as f:
        #         f.write("\n".join(imagepath_list))
        # testpath = Path(testpath)
        # imgpaths_file = os.path.join("img_paths", testpath.name + ".txt")
        # if os.path.exists(imgpaths_file):
        #     with open(imgpaths_file, "r") as f:
        #         imagepath_list = [l.strip() for l in f.readlines()]
        # else:
        #     imagepath_list = glob(str(testpath / '**/images-512/*.jpg'), recursive=True)
        #     with open(imgpaths_file, "w") as f:
        #         f.write("\n".join(imagepath_list))

        if self.frame_file is not None:
            testpath = Path(testpath)
            frames = np.loadtxt(self.frame_file, dtype=str)
            imagepath_list = [testpath / s / "images-512" / "cam_000.jpg" for s in frames]

        return imagepath_list


def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


if __name__ == "__main__":
    ds = NerSembleData("")
