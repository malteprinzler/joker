import numpy as np
import torch
from joker.third_party.MODNet.modnet_pipeline import MODNetPipeline
from PIL import Image

modnet_pipeline = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_modnet_pipeline():
    global modnet_pipeline
    if modnet_pipeline is None:
        modnet_pipeline = MODNetPipeline("assets/MODNet/modnet_photographic_portrait_matting.ckpt", device)

@torch.no_grad()
def modnet_mat_img(img: np.ndarray):
    global modnet_pipeline
    init_modnet_pipeline()
    matte = modnet_pipeline(Image.fromarray(img))
    matte = (matte[0,0].detach().cpu()*255).round().clip(0, 255).numpy().astype(np.uint8)
    return matte