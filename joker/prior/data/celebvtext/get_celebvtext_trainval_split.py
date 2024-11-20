import argparse
import copy
import multiprocessing
from pathlib import Path
import json
from textwrap import indent

import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('10.1.5.127', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)


def load_subj_arcface_embedding(subj, root):
    with open(root / subj / "arcface_embedding.txt", "r") as f:
        embedding = json.load(f)["embedding"]
    return embedding


if __name__ == "__main__":
    """
    splits celebvtext metas into train and validation sets based on distance of the subject arcface embeddings (makes sure that cosine similarity of val samples and their closest training sample is < than threshold)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--divex_file", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K/VALID_DIVEX_METAS.json")
    parser.add_argument("--out_root", type=Path, default="data/CELEBV_TEXT/PROCESSED_FRAMES_50K")
    parser.add_argument("--train_val_ratio", type=float, default=0.9)
    parser.add_argument("--id_thr", type=float, default=0.4)
    args = parser.parse_args()

    # root = Path("")
    # frame_file = root / "VALID_FLAME_FRAMES.txt"
    # out_train_seq_file = root / "TRAIN_SEQUENCES.txt"
    # out_val_seq_file = root / "VAL_SEQUENCES.txt"
    # out_val_seqgroup_file = root / "VAL_SEQUENCE_GROUPS.json"
    # train_val_ratio = .9
    # id_thr = 0.4

    # loading metas, frames, and subject embeddings
    with open(args.divex_file, "r") as f:
        all_metas = json.load(f)
    all_frames = [m["path"] for m in all_metas]
    all_subjs = np.array(list(sorted(np.unique([f.split("/")[0] for f in all_frames]))), dtype=str)

    with multiprocessing.Pool(32) as p:
        subj_arcface_emb = list(tqdm.tqdm(p.imap(partial(load_subj_arcface_embedding, root=args.root), all_subjs), total=len(all_subjs), desc="loading arcface embeddings"))

    subj_arcface_emb = np.array(subj_arcface_emb)

    # determining nr of validation subjects
    nsubjs = len(all_subjs)
    n_valsubjs = int(nsubjs * (1 - args.train_val_ratio))
    val_subj_mask = np.zeros(nsubjs, dtype=bool)


    def add_val_subj(idx):
        """
        adds subject of index `idx` to validation set, checks for duplicates in remaining dataset and if any duplicates are found, also copies them over to the validation set.
        works recursively, so also checks for duplicates of the detected duplicates ...
        """
        if not val_subj_mask[idx]:
            val_subj_mask[idx] = True

            current_arcface_emb = subj_arcface_emb[idx]
            compare_arcface_emb = subj_arcface_emb[~val_subj_mask]
            compare_idcs = np.arange(nsubjs)[~val_subj_mask]

            id_scores = np.sum(current_arcface_emb[None] * compare_arcface_emb, axis=1)

            # find duplicates
            sorted_idcs = np.argsort(id_scores)[::-1]
            sorted_compare_idcs = compare_idcs[sorted_idcs]
            n_duplicates = (id_scores > args.id_thr).astype(int).sum()

            # # visualize matches > id_thr and match scores
            # current_subj = subjs[idx]
            # compare_subjs = subjs[~val_subj_mask]
            # sorted_scores = id_scores[sorted_idcs]
            # sorted_subjs = [compare_subjs[i] for i in sorted_idcs]
            # ncols = int(np.ceil(np.sqrt(n_duplicates + 1)))
            # fig, axes = plt.subplots(ncols=ncols, nrows=ncols, figsize=(20, 20))
            # fig.suptitle(F"{n_duplicates} duplicates for subj {current_subj}")
            # axes = axes.flatten()
            # for i in range(n_duplicates):
            #     subj = sorted_subjs[i]
            #     id_score = sorted_scores[i]
            #     with open(root / subj / "arcface_embedding.txt", "r") as f:
            #         subj_img_path = root / json.load(f)["img_path"]
            #     axes[i + 1].imshow(Image.open(subj_img_path))
            #     t = axes[i + 1].text(0., 0., f"match: {subj}; {id_score:.3f}", transform=axes[i + 1].transAxes, fontsize=12)
            #     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
            # with open(root / current_subj / "arcface_embedding.txt", "r") as f:
            #     current_img_path = root / json.load(f)["img_path"]
            # axes[0].imshow(Image.open(current_img_path))
            # t = axes[0].text(0., 0., f"subj: {current_subj}", transform=axes[0].transAxes, fontsize=12)
            # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
            # [ax.axis("off") for ax in axes]
            # plt.tight_layout()
            # plt.show()
            # plt.close(fig)

            # also adding duplicates to val samples (including their duplicates):
            for i in range(n_duplicates):
                dupl_idx = sorted_compare_idcs[i]
                if not val_subj_mask[dupl_idx]:
                    add_val_subj(dupl_idx)
            return [all_subjs[idx]] + [all_subjs[sorted_compare_idcs[i]] for i in range(n_duplicates)]
        else:
            return list()


    new_val_subj_idx = nsubjs - 1
    pbar = tqdm.tqdm(total=n_valsubjs, desc="Getting Val Subjects")
    val_subject_groups = []
    while np.sum(val_subj_mask.astype(int)) < n_valsubjs:
        added_subjects = add_val_subj(new_val_subj_idx)
        if len(added_subjects) > 0:
            pbar.update(len(added_subjects))
            val_subject_groups.append(added_subjects)
        new_val_subj_idx -= 1

    val_subjs = np.sort(all_subjs[val_subj_mask])
    train_subjs = np.sort(all_subjs[~val_subj_mask])

    train_metas = [m for m in all_metas if m["path"].split("/")[0] in train_subjs]
    val_metas = [m for m in all_metas if m["path"].split("/")[0] in val_subjs]

    args.out_root.mkdir(exist_ok=True, parents=True)
    with open(args.out_root / "TRAIN_METAS.json", "w") as f:
        json.dump(train_metas, f, indent="\t")
    with open(args.out_root / "VAL_METAS.json", "w") as f:
        json.dump(val_metas, f, indent="\t")
    with open(args.out_root / "SUBJECT_SPLIT.json", "w") as f:
        json.dump(dict(train_subjs=train_subjs.tolist(), val_subjs=val_subjs.tolist()),
                  f, indent="\t")
