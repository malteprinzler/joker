import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .src.models.modnet import MODNet


class MODNetPipeline(nn.Module):
    ref_size = 512

    def __init__(self, ckpt_path, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.modnet = MODNet(backbone_pretrained=False)

        # loading and fixing pretrained state dict
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            k_new = k[7:]  # replacing "module." prefix of pretrained state dict
            fixed_state_dict[k_new] = v

        self.modnet.load_state_dict(fixed_state_dict)
        self.modnet = self.modnet.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def forward(self, im):
        # im either PIL Image or torch tensor of shape [N, 3, H, W] with values -1 ... 1

        if isinstance(im, Image.Image):
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
            im = self.transform(im)
        im = im.to(self.device)

        # add mini-batch dim
        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        # resize image for input

        im_b, im_c, im_h, im_w = im.shape

        if im_h != self.ref_size or im_w != self.ref_size:
            if max(im_h, im_w) < self.ref_size or min(im_h, im_w) > self.ref_size:
                if im_w >= im_h:
                    im_rh = self.ref_size
                    im_rw = int(im_w / im_h * self.ref_size)
                elif im_w < im_h:
                    im_rw = self.ref_size
                    im_rh = int(im_h / im_w * self.ref_size)
            else:
                im_rh = im_h
                im_rw = im_w

            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self.modnet(im, True)

        # resize and save matte
        if im_h != self.ref_size or im_w != self.ref_size:
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

        return matte


if __name__ == '__main__':
    pipeline = MODNetPipeline("pretrained/modnet_photographic_portrait_matting.ckpt")
    img = Image.open("")
    matte = pipeline(img)
    print()
