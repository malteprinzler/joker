import argparse
import json
from pathlib import Path
import numpy as np
import tqdm
from PIL import Image
import torch
import copy
from itertools import product

from joker.distillation.data.preprocess.utils import align_camera_parameters
from joker.prior.data.preprocess.deep3dfacerecon import predict_bfm_from_img, deep3dface
from joker.prior.data.preprocess.blip import predict_img_caption, naive_caption_reenactment
from joker.prior.data.preprocess.modnet import modnet_mat_img


def init_cam_sweep(ref_deep3dfacerecon, targ_deep3dfacerecon, prompt, out_dir, scale_factor):
    ref_head_coeffs = ref_deep3dfacerecon["head_coeffs"]
    targ_head_coeffs = targ_deep3dfacerecon["head_coeffs"]
    head_coeffs = copy.deepcopy(targ_head_coeffs)
    head_coeffs["id"] = ref_head_coeffs["id"]
    persc_proj = torch.tensor(head_coeffs["facemodel_perc_proj"], device=deep3dface.model.device)
    ndc_proj = torch.tensor(head_coeffs["ndc_proj"], device=deep3dface.model.device)
    targ_cam_params = deep3dface.headcoeff2camparams(head_coeffs, scale_factor=scale_factor)
    targ_mask = modnet_mat_img(targ_deep3dfacerecon["img"])

    out_target_img_path = out_dir / "target_img" / "target_img.jpg"
    out_target_cam_path = out_dir / "target_cam-512" / "target_cam.json"
    out_target_mask_path = out_dir / "target_mask" / "target_mask.png"
    out_ref_img_path = out_dir / "ref_img" / "ref_img.jpg"
    blip_caption_path = out_dir / "blip_captions" / f"cam_000.txt"

    out_target_img_path.parent.mkdir(exist_ok=True, parents=True)
    out_target_cam_path.parent.mkdir(exist_ok=True, parents=True)
    out_target_mask_path.parent.mkdir(exist_ok=True, parents=True)
    out_ref_img_path.parent.mkdir(exist_ok=True, parents=True)
    blip_caption_path.parent.mkdir(exist_ok=True, parents=True)

    Image.fromarray(targ_deep3dfacerecon["img"]).save(out_target_img_path)
    Image.fromarray(ref_deep3dfacerecon["img"]).save(out_ref_img_path)
    Image.fromarray(targ_mask).save(out_target_mask_path)
    with open(out_target_cam_path, "w") as f:
        json.dump(targ_cam_params, f)
    with open(blip_caption_path, "w") as f:
        f.write(prompt)

    return head_coeffs, persc_proj, ndc_proj


def render_cam_sweep(out_dir, head_coeffs, angles, persc_proj, ndc_proj, h, w, scale_factor, relative_to_input_pose):
    '''
    camera convention:
    - intrinsics: +x = right, +y = down, (0,0) lies at top left corner of top left pixel
    - extrinsics: +x = right, +y = down, +z = look-at
    - world: +x = left face side, +y = up, +z = towards camera rig

    '''
    face_recon = deep3dface.model

    for j, (az, el) in enumerate(tqdm.tqdm(angles, total=len(angles))):
        head_coeffs_ = copy.deepcopy(head_coeffs)

        # adding batch dimension to head_coeffs_
        for k in ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']:
            head_coeffs_[k] = [head_coeffs_[k]]

        phi = np.array(0, dtype=np.float32)
        if relative_to_input_pose:
            el += np.array(head_coeffs_["angle"][0][0], dtype=np.float32)
            az += np.array(head_coeffs_["angle"][0][1], dtype=np.float32)
            phi += np.array(head_coeffs_["angle"][0][2], dtype=np.float32)
        head_coeffs_["angle"] = [[el, az, phi]]

        # setting rendering settings to original state in case they were changed within this for loop
        face_recon.facemodel.persc_proj = persc_proj
        face_recon.renderer.rasterize_size = (h, w)
        face_recon.renderer.ndc_proj = ndc_proj

        face_recon_coeffs = face_recon.facemodel.unsplit_coeff(head_coeffs_, tensor_kwargs=dict(device=deep3dface.model.device))
        pred_vertex, pred_tex, pred_color, pred_lm, face_norm_roted = face_recon.facemodel.compute_for_render(face_recon_coeffs)
        pred_mask, _, pred_normal = face_recon.renderer(pred_vertex, face_recon.facemodel.face_buf, feat=face_norm_roted)
        normal_img = Image.fromarray(np.clip(np.round(pred_normal[0].cpu().permute(1, 2, 0).numpy() * 127.5 + 127.5), a_min=0, a_max=255).astype(np.uint8))

        # align image
        ndc_proj_cropped, persc_proj_cropped, normal_img_cropped = align_camera_parameters(pred_lm[0].cpu().numpy(), ndc_proj, persc_proj, h, w, im=normal_img)
        face_recon.facemodel.persc_proj = persc_proj_cropped
        face_recon.renderer.ndc_proj = ndc_proj_cropped
        head_coeffs_["facemodel_perc_proj"] = persc_proj_cropped.cpu().tolist()
        head_coeffs_["ndc_proj"] = ndc_proj_cropped.cpu().tolist()
        pred_vertex_cropped, pred_tex_cropped, pred_color_cropped, pred_lm_cropped, face_norm_roted_cropped = face_recon.facemodel.compute_for_render(face_recon_coeffs)
        pred_mask_cropped, _, pred_normal_cropped = face_recon.renderer(pred_vertex_cropped, face_recon.facemodel.face_buf, feat=face_norm_roted_cropped)
        normal_img_rendercropped = np.clip(np.round(((pred_normal_cropped * .5 + .5) * pred_mask_cropped)[0].cpu().permute(1, 2, 0).numpy() * 255), a_min=0, a_max=255).astype(np.uint8)
        cam_params = deep3dface.headcoeff2camparams(head_coeffs_, scale_factor=scale_factor)

        # # visualize cropped rerendered landmarks and normals
        # plt.imshow(normal_img_rendercropped)
        # plt.scatter(pred_lm_cropped[0, :, 0].cpu(), 511 - pred_lm_cropped[0, :, 1].cpu(), color="yellow")
        # plt.show()

        # # visualizing images before and after cropping
        # normal_img_rendercropped.save(out_dir / "rendercropped.jpg")
        # normal_img.save(out_dir / "uncropped.jpg")
        # normal_img_cropped.save(out_dir / "cropped.jpg")

        head_coeffs_["angle"] = [[float(x) for x in head_coeffs_["angle"][0]]]  # casting angles to float64 for json dumping

        camname = f"cam_{j:03d}"
        outpath_headcoeffs = out_dir / "bfm_eg3d" / (camname + ".json")
        outpath_normals = out_dir / "bfm_normals-512" / (camname + ".jpg")
        outpath_cam = out_dir / "cam-512" / (camname + ".json")

        outpath_headcoeffs.parent.mkdir(exist_ok=True, parents=True)
        outpath_normals.parent.mkdir(exist_ok=True, parents=True)
        outpath_cam.parent.mkdir(exist_ok=True, parents=True)

        with open(outpath_headcoeffs, "w") as f:
            json.dump(head_coeffs_, f, indent="\t")
        with open(outpath_cam, "w") as f:
            json.dump(cam_params, f, indent="\t")
        Image.fromarray(normal_img_rendercropped).save(outpath_normals)


def create_cam_sweep_grid_reenact(ref_deep3dfacerecon: dict, targ_deep3dfacerecon: dict, prompt: str, out_dir: Path, az_range, el_range, n_h, n_v, scale_factor=1.2 / 10, relative_to_input_pose=False):
    out_dir = Path(out_dir)

    head_coeffs, persc_proj, ndc_proj = init_cam_sweep(ref_deep3dfacerecon, targ_deep3dfacerecon, prompt, out_dir, scale_factor)
    h, w = targ_deep3dfacerecon["head_coeffs"]["hw_orig"]

    # getting grid azs and els
    az_angles = np.linspace(az_range[0], az_range[1], n_h, dtype=np.float32) * np.pi / 180
    el_angles = np.linspace(el_range[0], el_range[1], n_v, dtype=np.float32) * np.pi / 180
    angles = list(product(az_angles, el_angles))

    # rendering and saving normal maps / camera parameters ...
    render_cam_sweep(out_dir, head_coeffs, angles, persc_proj, ndc_proj, h, w, scale_factor, relative_to_input_pose)


def create_cam_sweep_spiral_reenact(ref_deep3dfacerecon: dict, targ_deep3dfacerecon: dict, prompt: str, out_dir: Path, az_range, el_range, nframes, nrotations, nmaxamplituderotations=1, scale_factor=1.2 / 10, relative_to_input_pose=False):
    assert (nrotations - nmaxamplituderotations) % 2 == 0  # need even increasing and decreasing of normals

    out_dir = Path(out_dir)

    h, w = targ_deep3dfacerecon["head_coeffs"]["hw_orig"]
    head_coeffs, persc_proj, ndc_proj = init_cam_sweep(ref_deep3dfacerecon, targ_deep3dfacerecon, prompt, out_dir, scale_factor)

    # getting grid azs and els
    t = np.linspace(0, 1, nframes + 1, dtype=np.float32)[:-1]
    phi = 2 * np.pi * t * nrotations
    period_time = 1 / nrotations
    if nrotations == nmaxamplituderotations:
        r = np.ones_like(t)
    else:
        tstart_maxamplituderotation = period_time * ((nrotations - nmaxamplituderotations) // 2)
        tend_maxamplituderotation = tstart_maxamplituderotation + nmaxamplituderotations * period_time
        r = np.minimum(np.minimum(t / tstart_maxamplituderotation, np.ones_like(t)), 1 - (t - tend_maxamplituderotation) / tstart_maxamplituderotation)

    x = r * np.cos(phi)  # normalized from -1 ... +1
    y = r * np.sin(phi)  # normalized from -1 ... +1

    az_angles = (x * .5 + .5) * (az_range[1] - az_range[0]) + az_range[0]  # renormalizing from -1 ... 1 -> az_min ... az_max
    el_angles = (y * .5 + .5) * (el_range[1] - el_range[0]) + el_range[0]

    # # visualize angles
    # plt.figure()
    # plt.plot(az_angles, label="az")
    # plt.plot(el_angles, label="el")
    # plt.legend(loc="best")
    # plt.show()

    az_angles *= np.pi / 180
    el_angles *= np.pi / 180
    angles = list(zip(az_angles, el_angles))

    # rendering and saving normal maps / camera parameters ...
    render_cam_sweep(out_dir, head_coeffs, angles, persc_proj, ndc_proj, h, w, scale_factor, relative_to_input_pose)


def generate_crossreenact_distillation_dataset(
        ref_path="",
        targ_path="",
        out_root="",
        az_train=45,
        el_train=20,
        az_val=None,
        el_val=0,
        n_h_train=20,
        n_v_train=20,
        n_h_val=15,
        n_v_val=1,
        nspiralframes=300,
        nmaxamplitudespiralrotations=1,
        nspiralrotations=3):
    """
    generates full distillation (and validation) datasets for joker 3D distillation from a reference image and a target image
    """

    out_root = Path(out_root)
    ref_path = Path(ref_path)
    targ_path = Path(targ_path)

    # preprocessing ref and targ images
    deep3dfacerecon_results_ref = predict_bfm_from_img(ref_path, crop=True)
    deep3dfacerecon_results_targ = predict_bfm_from_img(targ_path, crop=True)

    caption_ref = predict_img_caption(deep3dfacerecon_results_ref["img"])
    caption_targ = predict_img_caption(deep3dfacerecon_results_targ["img"])
    reenacted_caption = naive_caption_reenactment(caption_ref, caption_targ, raise_error=True)

    # getting angles right
    if az_val is None:
        az_val = az_train
    if el_val is None:
        el_val = el_train
    az_min_train = -1 * az_train
    az_max_train = az_train
    el_min_train = -1 * el_train
    el_max_train = el_train
    az_min_val = -1 * az_val
    az_max_val = az_val
    el_min_val = -1 * el_val
    el_max_val = el_val

    create_cam_sweep_grid_reenact(deep3dfacerecon_results_ref,
                                  deep3dfacerecon_results_targ,
                                  prompt=reenacted_caption,
                                  out_dir=out_root / "sequences" / "CAM_SWEEP" / "frame_00000",
                                  az_range=(az_min_train, az_max_train),
                                  el_range=(el_min_train, el_max_train),
                                  n_h=n_h_train,
                                  n_v=n_v_train)

    create_cam_sweep_grid_reenact(deep3dfacerecon_results_ref,
                                  deep3dfacerecon_results_targ,
                                  prompt=reenacted_caption,
                                  out_dir=out_root / "sequences" / "VAL" / "frame_00000",
                                  az_range=(az_min_val, az_max_val),
                                  el_range=(el_min_val, el_max_val),
                                  n_h=n_h_val,
                                  n_v=n_v_val)

    create_cam_sweep_spiral_reenact(deep3dfacerecon_results_ref,
                                    deep3dfacerecon_results_targ,
                                    prompt=reenacted_caption,
                                    out_dir=out_root / "sequences" / "SPIRAL" / "frame_00000",
                                    az_range=(az_min_train, az_max_train),
                                    el_range=(el_min_train, el_max_train),
                                    nframes=nspiralframes,
                                    nrotations=nspiralrotations, nmaxamplituderotations=nmaxamplitudespiralrotations,)



if __name__ == "__main__":
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.205', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=Path)
    parser.add_argument("--targ_path", type=Path)
    parser.add_argument("--out_root", type=Path)
    parser.add_argument("--az_train", type=float, default=45.)
    parser.add_argument("--el_train", type=float, default=20.)
    parser.add_argument("--nspiralframes", type=int, default=300)
    parser.add_argument("--nmaxamplitudespiralrotations", type=int, default=1)
    parser.add_argument("--nspiralrotations", type=int, default=3)
    args = parser.parse_args()

    # inputs: ref_img_path (uncropped), targ_img (uncropped)
    generate_crossreenact_distillation_dataset(
        ref_path=args.ref_path,
        targ_path=args.targ_path,
        out_root=args.out_root,
        az_train=args.az_train,
        el_train=args.el_train,
        nspiralframes=args.nspiralframes,
        nmaxamplitudespiralrotations=args.nmaxamplitudespiralrotations,
        nspiralrotations=args.nspiralrotations,
    )
