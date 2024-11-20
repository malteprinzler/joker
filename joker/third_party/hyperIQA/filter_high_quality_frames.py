from pathlib import Path
import numpy as np
import tqdm
import multiprocessing

root = Path("")
in_frame_file = root / "VALID_FLAME_FRAMES.txt"
out_frame_file = root / "VALID_FLAME_FRAMES_HQ.txt"
threshold = 40.
nworkers = 5

frames = [str(f) for f in np.loadtxt(in_frame_file, dtype=str)]


def check_frame_hq(frame):
    score_file = root / frame / "hyperiqa_score" / "cam_00.txt"
    with open(score_file, "r") as f:
        score = float(f.readline().strip())
    if score >= threshold:
        return True
    else:
        # print(f"Dropped frame {frame} with quality score: {score:.1f}.")
        return False


if __name__ == "__main__":
    with multiprocessing.Pool(nworkers) as p:
        hq_mask = list(tqdm.tqdm(p.imap(check_frame_hq, frames), total=len(frames)))
    frames = np.array(frames, dtype=str)
    hq_mask = np.array(hq_mask)
    hq_frames = frames[hq_mask]
    hq_frames = list(hq_frames)
    with open(out_frame_file, "w") as f:
        f.write("\n".join(hq_frames))
    print(f"Written {len(hq_frames)} of {len(frames)} ({len(hq_frames) / len(frames) * 100:.1f}%) to {out_frame_file}.")
