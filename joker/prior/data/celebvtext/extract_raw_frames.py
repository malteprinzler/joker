import argparse
import multiprocessing
from pathlib import Path
import os
import numpy as np
import cv2

import tqdm
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image


def extract_video_frames(io, noutframes):
    video_fp, out_dir = io
    tmp_out_dir = Path("/tmp/CelebVText/") / out_dir.name
    try:
        vidcap = cv2.VideoCapture(str(video_fp))
        nframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        sample_frames = np.round(np.linspace(0, nframes - 1, noutframes + 2)[1:-1]).astype(int)

        for frame_idx in sample_frames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vidcap.read()
            tmp_outpath = tmp_out_dir / "sequences" / "SEQ-1" / f"frame_{frame_idx:05d}" / "images-512" / "cam_00.jpg"
            tmp_outpath.parent.mkdir(exist_ok=True, parents=True)
            img = Image.fromarray(frame[:, :, ::-1])
            img.save(str(tmp_outpath))

        with open(tmp_out_dir / "original_vid.txt", "w") as f:
            f.write(video_fp.name)

        vidcap.release()
        del vidcap

        out_dir.parent.mkdir(exist_ok=True, parents=True)
        os.system(f"cp -r {tmp_out_dir} {out_dir.parent}")
        os.system(f"rm -r {tmp_out_dir}")

    except Exception as e:
        print(f"ERROR with {video_fp}: {e}")


if __name__ == "__main__":
    """
    extract frames from downloaded celebvtext videos
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", default="data/CELEBV_TEXT/VIDEOS", type=Path)
    parser.add_argument("--out_root", default="data/CELEBV_TEXT/RAW_FRAMES_50K", type=Path)
    parser.add_argument("--frames_per_vid", default=10, type=int)
    parser.add_argument("--nworkers", default=10, type=int)
    parser.add_argument("--vid_list_file", default="data/CELEBV_TEXT/50K_BALANCED_VIDEO_FILES.txt", type=Path)
    args = parser.parse_args()

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    vid_name_list = np.loadtxt(str(args.vid_list_file), dtype=str, ndmin=1)
    src_files = [args.in_root / vid_name for vid_name in vid_name_list]
    multiprocessing.set_start_method("spawn")

    # checking if all src_files exist
    for f in tqdm.tqdm(src_files, total=len(src_files), desc="Checking src file existence"):
        if not f.exists():
            raise FileNotFoundError(f"{f} is specified in vid_list_file {args.vid_list_file} but does not exist! Please recreate this file using the get_balanced_vidfileslist.py ")

    my_idcs = np.array_split(np.arange(len(src_files)), njobs)[jobid]
    my_out_dirs = [args.out_root / f"{idx:05d}" for idx in my_idcs]
    my_src_files = [src_files[idx] for idx in my_idcs]
    nmysrc = len(my_src_files)
    print(f"Processing {len(my_idcs)} out of {len(src_files)} files:")
    worker_fn = partial(extract_video_frames, noutframes=args.frames_per_vid)

    # for io in tqdm.tqdm(zip(my_src_files, my_out_dirs), "Sequences", leave=False):
    #     worker_fn(io)

    with multiprocessing.Pool(args.nworkers) as p:
        list(tqdm.tqdm(p.imap_unordered(worker_fn, zip(my_src_files, my_out_dirs)), total=nmysrc, desc="Processing Videos", leave=True))
