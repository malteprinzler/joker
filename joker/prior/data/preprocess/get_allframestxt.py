import argparse
from pathlib import Path
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data/CELEBV_TEXT/RAW_FRAMES_50K", type=Path)
args = parser.parse_args()

root = args.root
outfile = root / "ALL_FRAMES.txt"

all_frame_dirs = list()

subj_dirs = sorted([p for p in root.iterdir() if p.name.isnumeric() and p.is_dir()])
for subj_dir in tqdm.tqdm(subj_dirs, "Subjects"):
    sequence_dirs = sorted([p for p in (subj_dir / "sequences").iterdir() if p.is_dir()])
    for seq_dir in tqdm.tqdm(sequence_dirs, "Sequences", leave=False):
        frame_dirs = sorted([p for p in seq_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")])
        frame_dirs = [str(p.relative_to(root)) for p in frame_dirs]
        all_frame_dirs = all_frame_dirs + frame_dirs

with open(outfile, "w") as f:
    f.write("\n".join(all_frame_dirs))
print(f"Written {len(all_frame_dirs)} frames to {outfile}")
