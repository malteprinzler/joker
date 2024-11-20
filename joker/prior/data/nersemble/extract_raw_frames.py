import argparse
import random
import multiprocessing
from pathlib import Path
import os
import numpy as np
import cv2

import tqdm
from PIL import Image
from functools import partial


def extract_video_frames(video_fp, out_root, nframes):
    try:
        identity, sequence, cam = video_fp.parents[1].name, video_fp.parents[0].name, video_fp.stem
        vidcap = cv2.VideoCapture(str(video_fp))
        nsrcframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        sample_frames = np.round(np.linspace(0, nsrcframes - 1, nframes + 2)[1:-1]).astype(int)

        for frame_idx in sample_frames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vidcap.read()
            outpath = out_root / identity / "sequences" / sequence / f"frame_{frame_idx:05d}" / "images-512" / (cam + ".jpg")
            outpath.parent.mkdir(exist_ok=True, parents=True)
            image = Image.fromarray(frame[:, :, ::-1])
            image.save(str(outpath))
        vidcap.release()
        del vidcap

    except Exception as e:
        print(f"ERROR with {video_fp}: {e}")


def process_sequence_folder(sequence_folder, out_root, frames_per_vid):
    try:
        tmp_out_root = Path("/tmp/NERSEMBLE")
        identity, sequence = sequence_folder.parent.name, sequence_folder.name
        in_path = tmp_out_root / identity / "sequences" / sequence
        outpath = out_root / identity / "sequences"
        if in_path.exists():
            os.system(f"rm -r {in_path}")
        video_files = sorted([p for p in sequence_folder.iterdir() if p.suffix.lower() == ".mp4" and p.name.startswith("cam_")])
        for file in video_files:
            extract_video_frames(file, out_root=tmp_out_root, nframes=frames_per_vid)
        outpath.mkdir(exist_ok=True, parents=True)
        os.system(f"cp -r {in_path} {outpath}")
        os.system(f"rm -r {in_path}")
    except Exception as e:
        print(f"ERROR with sequence folder {sequence_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw frames from the nersemble dataset with an emphasis on sequences with extreme expressions")
    parser.add_argument("--in_root", type=Path, default="data/NERSEMBLE/VIDEOS")
    parser.add_argument("--out_root", type=Path, default="data/NERSEMBLE/RAW_FRAMES_EXPR_10K")
    parser.add_argument("--frames_per_vid", type=int, default=4)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--sequences", type=str, nargs="*", default=[
        "EMO-1-shout+laugh",
        "EMO-3-angry+sad",
        "EMO-4-disgust+happy",
        "EXP-2-eyes",
        "EXP-3-cheeks+nose",
        "EXP-4-lips",
        "EXP-5-mouth",
        "EXP-6-tongue-1",
        "EXP-7-tongue-2",
        "EXP-8-jaw-1",
        "EXP-9-jaw-2",
        "FREE",
    ])
    args = parser.parse_args()

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    src_folders = sorted([p for p in args.in_root.iterdir() if p.is_dir() and p.name.isnumeric()])
    my_src_folders = np.array_split(src_folders, njobs)[jobid]
    print(f"Processing {len(my_src_folders)} out of {len(src_folders)} files:")
    multiprocessing.set_start_method("spawn")

    for src_folder in tqdm.tqdm(my_src_folders, "Identity"):
        random.seed(src_folder.name)
        out_folder = args.out_root / src_folder.name
        sequence_folders = [src_folder / s for s in args.sequences]
        worker_fn = partial(process_sequence_folder, out_root=args.out_root, frames_per_vid=args.frames_per_vid)

        # for folder in tqdm.tqdm(sequence_folders, "Sequences", leave=False):
        #     worker_fn(folder)

        with multiprocessing.Pool(args.n_workers) as p:
            list(tqdm.tqdm(p.imap_unordered(worker_fn, sequence_folders), total=len(sequence_folders), desc="Sequences", leave=False))

        os.system(f"cp {src_folder / 'camera_params.json'} {out_folder}")
        os.system(f"cp -r {src_folder / 'BACKGROUND'} {out_folder}")
