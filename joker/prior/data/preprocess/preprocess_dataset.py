import argparse
from joker.utils import str2bool
import json
from pathlib import Path
import numpy as np
import tqdm
from functools import partial
from PIL import Image
import os
from joker.prior.data.preprocess.deep3dfacerecon import predict_bfm_from_img
from joker.prior.data.preprocess.blip import predict_img_caption
from joker.prior.data.preprocess.modnet import modnet_mat_img
from joker.prior.data.preprocess.hyperiqa import get_img_hyperiqa
from joker.prior.data.preprocess.emica import flame_from_img


def process_frame(frame_dir: Path, out_root: Path, blur_pad=False):
    img_dir = frame_dir / "images-512"
    frame, sequence, _, subject = [p.name for p in list(img_dir.parents)[:4]]
    in_subject_dir = frame_dir.parents[2]
    out_subject_dir = out_root / subject
    out_frame_dir = out_subject_dir / "sequences" / sequence / frame
    out_frame_dir.mkdir(exist_ok=True, parents=True)
    os.system(f"cp {in_subject_dir / 'original_vid.txt'} {out_subject_dir / 'original_vid.txt'}")

    for img_path in [f for f in img_dir.iterdir() if f.name.startswith("cam_") and f.name.endswith(".jpg")]:
        ### cropping and bfm estimation
        deep3dfacerecon_results = predict_bfm_from_img(img_path, blur_pad=blur_pad, crop=True)
        out_img_file = out_frame_dir / ("images-512") / (img_path.stem + ".jpg")
        out_head_file = out_frame_dir / "bfm_eg3d" / (img_path.stem + ".json")
        out_normal_file = out_frame_dir / "bfm_normals-512" / (img_path.stem + ".jpg")
        out_img_file.parent.mkdir(exist_ok=True, parents=True)
        out_head_file.parent.mkdir(exist_ok=True, parents=True)
        out_normal_file.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(deep3dfacerecon_results["img"]).save(str(out_img_file))
        Image.fromarray(deep3dfacerecon_results["normal"]).save(str(out_normal_file))
        with open(out_head_file, "w") as f:
            json.dump(deep3dfacerecon_results["head_coeffs"], f, indent="\t")

        ### captioning
        caption = predict_img_caption(deep3dfacerecon_results["img"])
        out_caption_file = out_frame_dir / "blip_captions" / (img_path.stem + ".txt")
        out_caption_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_caption_file, "w") as f:
            f.write(caption)

        ### fg-bg segmentation
        mask = modnet_mat_img(deep3dfacerecon_results["img"])
        out_mask_file = out_frame_dir / "mask-512" / (img_path.stem + ".png")
        out_mask_file.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(mask).save(str(out_mask_file))

        ### hyperiqa score
        hyperiqa_score = get_img_hyperiqa(deep3dfacerecon_results["img"])
        out_hyperiqa_file = out_frame_dir / "hyperiqa_score" / (img_path.stem + ".txt")  # lmks on original image
        out_hyperiqa_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_hyperiqa_file, "w") as f:
            f.write(str(hyperiqa_score))

        ### flame prediction
        flame_results = flame_from_img(deep3dfacerecon_results["img"])
        out_flame_normal_file = out_frame_dir / "flame_normals-512" / (img_path.stem + ".jpg")
        out_flame_param_file = out_frame_dir / "flame_params-512" / (img_path.stem + ".json")
        out_flame_normal_file.parent.mkdir(exist_ok=True, parents=True)
        out_flame_param_file.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(flame_results["normal_image"]).save(str(out_flame_normal_file))
        with open(out_flame_param_file, "w") as f:
            json.dump(flame_results["flame_params"], f, indent="\t")


if __name__ == "__main__":
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.127', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=Path, default="data/CELEBV_TEXT/RAW_FRAMES_50K")
    parser.add_argument("--out_root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--frame_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/ALL_FRAMES.txt")
    parser.add_argument("--blur_pad", type=str2bool, default=True)
    parser.add_argument("--nworkers", type=int, default=0)
    args = parser.parse_args()

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    all_frames = np.loadtxt(args.frame_file, dtype=str)
    all_frames = [args.in_root / s for s in all_frames]
    my_frames = np.array_split(all_frames, njobs)[jobid]
    print(f"Processing {len(my_frames)} out of {len(all_frames)} frames:")
    [print(f) for f in my_frames]
    print()

    worker_fn = partial(process_frame, out_root=args.out_root, blur_pad=args.blur_pad)

    for folder in tqdm.tqdm(my_frames, "Frames", leave=False):
        worker_fn(folder)

    # multiprocessing.set_start_method("spawn")
    # with multiprocessing.Pool(nworkers) as p:
    #     list(tqdm.tqdm(p.imap_unordered(worker_fn, my_frames), total=len(my_frames), desc="Frames"))

    if jobid == 0:
        os.system(f"cp {args.frame_file} {args.out_root/'ALL_FRAMES.txt'}")
