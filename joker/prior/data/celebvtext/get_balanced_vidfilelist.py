import argparse
from pathlib import Path
from collections import defaultdict
import tqdm

"""
Subsamples X videos from CelebVText that stem from as diverse video sources as possible 
"""
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default="data/CELEBV_TEXT/VIDEOS")
parser.add_argument("--outpath", type=Path, default="data/CELEBV_TEXT/50K_BALANCED_VIDEO_FILES.txt")
parser.add_argument("--nvideos", type=int, default=50_000)
args = parser.parse_args()

vid_dict = defaultdict(list)
print("Constructing vid_dict")
for f in [p for p in args.root.iterdir() if p.name.endswith(".mp4")]:
    vid = f.name[:11]
    vid_dict[vid].append(f.name)

vids = sorted(list(vid_dict.keys()))
nvids = len(vids)

i = 0
pbar = tqdm.tqdm(list(range(args.nvideos)), "Processing")
video_list = list()
while len(video_list) < args.nvideos:
    vid = vids[i % nvids]
    vid_video_list = vid_dict[vid]
    if len(vid_video_list) > 0:
        video_list.append(vid_video_list.pop(0))
        pbar.update()
    i += 1
args.outpath.parent.mkdir(exist_ok=True, parents=True)
with open(args.outpath, "w") as f:
    f.write("\n".join(video_list))
print(f"Success. Wrote to {args.outpath}")
