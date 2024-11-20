import random

import torch
import torchvision
from joker.third_party.hyperIQA.models import HyperNet, TargetNet
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
import hashlib
from torch import multiprocessing
from functools import partial
import os


def seed_from_str(s):
    sha1 = hashlib.sha1()
    sha1.update(str.encode(s))
    hash_as_hex = sha1.hexdigest()
    # convert the hex back to int and restrict it to the relevant int range
    seed = int(hash_as_hex, 16) % 4294967295

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_hyper = None


def init_hyperiqa():
    global model_hyper
    if model_hyper is None:
        model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
        model_hyper.train(False)
        # load our pre-trained model on the koniq-10k dataset
        model_hyper.load_state_dict((torch.load('assets/hyperIQA/koniq_pretrained.pkl')))


@torch.no_grad()
def get_img_hyperiqa(img: np.ndarray, ncrops=10, seed_str=""):
    """
    img: np.ndarray H,W,3 0...255
    seed_str: str to calculate and set seed from (hyperiqa picks 10 'random' crops from img to calculate score)
    """
    init_hyperiqa()
    seed_from_str(seed_str)

    totensor = torchvision.transforms.ToTensor()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    # random crop 10 patches and calculate mean quality score
    pred_scores = []
    img = totensor(Image.fromarray(img)).to(device)
    assert img.max() <= 1 and img.min() >= 0

    for i in range(ncrops):
        img_crop = transforms(img)
        img_crop = torch.tensor(img_crop.cuda()).unsqueeze(0)
        paras = model_hyper(img_crop)  # 'paras' contains the network weights conveyed to target network
        model_target = TargetNet(paras)  # Building target network

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))
    score = np.mean(pred_scores)
    return score
