import random

import torch
import torchvision
import models
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


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# load our pre-trained model on the koniq-10k dataset
model_hyper.load_state_dict((torch.load('pretrained/koniq_pretrained.pkl')))
totensor = torchvision.transforms.ToTensor()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=224),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])


@torch.no_grad()
def score_img(img_path, ncrops=10):
    # random crop 10 patches and calculate mean quality score
    pred_scores = []
    img = totensor(pil_loader(img_path)).cuda()

    for i in range(ncrops):
        img_crop = transforms(img)
        img_crop = torch.tensor(img_crop.cuda()).unsqueeze(0)
        paras = model_hyper(img_crop)  # 'paras' contains the network weights conveyed to target network
        model_target = models.TargetNet(paras)  # Building target network

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))
    score = np.mean(pred_scores)
    return score


def process_frame(frame, root):
    img_path = frame / "images-512" / "cam_00.jpg"
    seed_from_str(str(img_path.relative_to(root)))

    quality_score = score_img(img_path)

    # writing results
    out_path = frame / "hyperiqa_score" / "cam_00.txt"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        f.write(str(quality_score))

    # # visualizing results
    # fig = plt.figure()
    # plt.imshow(Image.open(img_path))
    # plt.title(f"{quality_score:.2f}")
    # plt.show()
    # plt.close(fig)


if __name__ == "__main__":
    root = Path("")
    frame_file = root / "ALL_FRAMES.txt"
    frame_list = [root / f for f in np.loadtxt(frame_file, dtype=str)]
    nworkers = 4

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    my_frame_list = np.array_split(frame_list, njobs)[jobid]
    print(f"Processing {len(my_frame_list)} out of {len(frame_list)} files:")

    worker_fn = partial(process_frame, root=root)

    # for frame in tqdm.tqdm(my_frame_list):
    #     worker_fn(frame)

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(nworkers) as p:
        r = list(tqdm.tqdm(p.imap_unordered(worker_fn, my_frame_list), total=len(my_frame_list)))
