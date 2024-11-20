import argparse
from functools import partial
from joker.prior.evaluate import ArcFaceEvaluator
from pathlib import Path
from PIL import Image
import numpy as np
from torch import multiprocessing
import tqdm
import json


def get_samples_list_from_divex(meta_fp):
    with open(meta_fp, "r") as f:
        metas = json.load(f)

    # getting subject's least expressive sample
    least_expr_samples = dict()
    for meta in metas:
        subject = meta["path"].split("/")[0]
        if subject not in least_expr_samples:
            least_expr_samples[subject] = meta
        else:
            bl = least_expr_samples[subject]
            if bl["expressiveness"] > meta["expressiveness"]:
                least_expr_samples[subject] = meta

    subjects = sorted(least_expr_samples.keys())
    img_paths = [least_expr_samples[subj]["path"] for subj in subjects]
    return img_paths


def load_img(img_path):
    return np.array(Image.open(str(img_path)).convert("RGB"))


def process_img(img_path, root):
    out_path = root / img_path.parents[4].name / "arcface_embedding.txt"
    # print("load img")
    img = load_img(img_path)
    #     print("calc embedding")
    face = arcface.detect_face(img)
    if face is not None:
        embedding = face.normed_embedding.tolist()
    else:
        embedding = None

    with open(out_path, "w") as f:
        json.dump(dict(img_path=str(img_path.relative_to(root)), embedding=embedding), f, indent="\t")
    #     print("write")


#     print("finished writing")


arcface = ArcFaceEvaluator()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates and stores arcface feature for each subject based on the frame with smallest expression and stores it under <subject_root>/arcface_embedding.txt.")
    parser.add_argument("--root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--divex_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/VALID_DIVEX_METAS.json")
    parser.add_argument("--n_workers", type=int, default=10)
    args = parser.parse_args()

    # reading from divex
    all_imgs = get_samples_list_from_divex(args.divex_file)
    all_imgs = [args.root / f for f in all_imgs]

    worker_fn = partial(process_img, root=args.root)

    # for img_path in tqdm.tqdm(all_imgs):
    #     worker_fn(img_path)

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(args.n_workers) as p:
        list(tqdm.tqdm(p.imap_unordered(worker_fn, all_imgs), total=len(all_imgs)))
