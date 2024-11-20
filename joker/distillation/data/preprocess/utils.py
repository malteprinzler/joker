import copy
import numpy as np
import torch
from PIL import Image
from joker.prior.data.preprocess.deep3dfacerecon import deep3dface



def crop_persp_proj(persp_proj, bbx, h_in, w_in, h_out, w_out):
    """
    applying image space cropping to perspective camera projection matrix
    h, w: height, width of input / output image
    bbx: bounding box left, top, right, bottom on input image
    persp_proj: torch.tensor 3x3 (origin bottom left)
    """

    persp_proj = copy.deepcopy(persp_proj)

    # translation
    w_bbx = bbx[2] - bbx[0]
    h_bbx = bbx[3] - bbx[1]
    persp_proj[0, 2] -= bbx[0]
    persp_proj[1, 2] -= h_in - 1 - bbx[3]

    # resizing
    f = w_out / w_bbx, h_out / h_bbx  # note that persp proj is used for landmark projection and here bottm-left pixel center is considered 0,0 whereas resizing typically assumes bottom-left pixel center is 0.5,0.5
    trafo = torch.tensor([[f[0], 0, f[0] * 0.5 - 0.5],
                          [0, f[1], f[1] * 0.5 - 0.5],
                          [0, 0, 1]], dtype=persp_proj.dtype, device=persp_proj.device)
    persp_proj = trafo @ persp_proj
    return persp_proj


def crop_ndc_proj(ndc_proj, bbx, h_in, w_in, h_out, w_out):
    """
    cropping ndc projection matrix
    ndc_proj: ndc projection matrix, torch tensor 4x4,  (+1, +1 is top right and -1, -1 top left of image)
    """
    bbx_ndc = np.array([bbx[0] / w_in, bbx[1] / h_in, bbx[2] / w_in, bbx[3] / h_in]) * 2 - 1  # ndc coordinates normalized from -1 ... +1, origin top left

    mx = 2. / (bbx_ndc[2] - bbx_ndc[0])
    zx = 1. - mx * bbx_ndc[2]
    my = 2. / (bbx_ndc[3] - bbx_ndc[1])
    zy = 1 - my * bbx_ndc[3]

    crop_trafo = torch.tensor(
        [[mx, 0, 0, zx],
         [0, my, 0, zy],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        device=ndc_proj.device, dtype=ndc_proj.dtype
    )

    cropped_ndc_proj = crop_trafo @ ndc_proj
    return cropped_ndc_proj


def align_camera_parameters(lm, ndc_proj, persp_proj, h, w, im=None):
    """

    lm: np.ndarray 68x2 with origin bottom left
    """
    if im is None:
        im_np = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        im_np = np.array(im)
    lm = copy.deepcopy(lm)
    lm[:, -1] = h - 1 - lm[:, -1]

    im_crop, bbx_orig = deep3dface.crop_img(im_np, lm, return_bbx_orig=True)  # bounding box of aligned crop on original image
    h_crop, w_crop = im_crop.shape[:2]

    persc_proj_cropped = crop_persp_proj(persp_proj.T, bbx_orig, h, w, h_crop, w_crop).T  # note that deep3dfacerecon stores transposed persp projection matrices
    ndc_proj_cropped = crop_ndc_proj(ndc_proj, bbx_orig, h, w, h_crop, w_crop)

    outputs = [ndc_proj_cropped, persc_proj_cropped]

    if im is not None:
        outputs.append(Image.fromarray(im_crop))
    return outputs