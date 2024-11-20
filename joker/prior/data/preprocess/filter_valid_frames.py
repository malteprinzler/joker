import argparse
import random
from pathlib import Path
from functools import partial
import numpy as np
import tqdm
import json
from PIL import Image
import multiprocessing


def check_frame(frame_in: Path, lmk_margin):
    in_images = sorted([f for f in (frame_in / "images-512").iterdir() if f.name.startswith("cam_") and f.name.endswith(".jpg")])
    ret = True
    discard_reason = None
    for img in in_images:

        # checking flame param file
        flame_param_file = img.parents[1] / "flame_params-512" / img.name.replace(".jpg", ".json")
        if not flame_param_file.exists():
            ret = False
            discard_reason = "Flame param file doesnt exist"
            break

        with open(flame_param_file, "r") as f:
            flame_params = json.load(f)
        if flame_params["shapecode"] is None:
            ret = False
            discard_reason = "Flame shapecode is none"
            break
        if flame_params["cam_out"][0] < 6:
            ret = False
            discard_reason = "Flame cam zoom factor below 6"
            break

        # checking bfm file
        bfm_file = img.parents[1] / "bfm_eg3d" / img.name.replace(".jpg", ".json")
        if not bfm_file.exists():
            ret = False
            discard_reason = "BFM file doesnt exist"
            break
        with open(bfm_file, "r") as f:
            bfm_params = json.load(f)
        if bfm_params["id"] is None:
            ret = False
            discard_reason = "BFM id is none"
            break
        # checking lmks
        lmks_raw = np.array(bfm_params["lmks_uncropped"])
        H_raw, W_raw = bfm_params["hw_uncropped"]
        w_lmk_margin_px = int(W_raw * lmk_margin)
        h_lmk_margin_px = int(H_raw * lmk_margin)
        allowed_lmk_bbx = np.array([[w_lmk_margin_px, h_lmk_margin_px], [W_raw - w_lmk_margin_px, H_raw - h_lmk_margin_px]])
        if np.any(lmks_raw < allowed_lmk_bbx[0][None]) or np.any(lmks_raw > allowed_lmk_bbx[1][None]):
            ret = False
            discard_reason = "lmks out of bounds"
            break

        # checking hq score
        score_file = img.parents[1] / "hyperiqa_score" / img.name.replace(".jpg", ".txt")
        if not score_file.exists():
            ret = False
            discard_reason = "hyperiqa file doesnt exist"
            break
        else:
            with open(score_file, "r") as f:
                hq_score = float(f.readline().strip())
            if hq_score < 40:
                ret = False
                discard_reason = f"hq_score={hq_score:.1f} which is smaller than 40"
                break

    if not ret:
        print(f"\nInvalid Frame: {frame_in}\nDiscard Reason: {discard_reason}\n")

    return ret


if __name__ == "__main__":
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace('10.1.5.205', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--frame_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/ALL_FRAMES.txt")
    parser.add_argument("--out_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/VALID_FRAMES.txt")
    parser.add_argument("--lmk_margin", type=float, default=0.045, help="margin around image border that lmks should lie in")
    parser.add_argument("--n_workers", type=int, default=32)
    args = parser.parse_args()

    print("Root", args.root)
    print("in_frame_file", args.frame_file)
    print("out_file", args.out_file)

    frame_names = [str(s) for s in np.loadtxt(args.frame_file, dtype=str)]
    frame_paths = [args.root / s for s in frame_names]

    worker_fn = partial(check_frame, lmk_margin=args.lmk_margin)

    # valid_frame_mask = list()
    # for frame in tqdm.tqdm(frame_paths, "Frames"):
    #     valid_frame_mask.append(worker_fn(frame))

    with multiprocessing.Pool(args.n_workers) as p:
        valid_frame_mask = list(tqdm.tqdm(p.imap(worker_fn, frame_paths), "Frames", total=len(frame_paths)))

    valid_frames = [f for i, f in enumerate(frame_names) if valid_frame_mask[i]]

    with open(args.out_file, "w") as f:
        f.write("\n".join(s for s in valid_frames))

    print(f"Done. {len(valid_frames)} of {len(frame_names)} frames remaining ({len(valid_frames) / len(frame_names) * 100:.1f}%). Written to {args.out_file}. ")
