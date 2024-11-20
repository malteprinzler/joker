import numpy as np
import argparse
import json
from pathlib import Path

NERSEMBLE_VAL_IDS = ['316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '329', '330', '331', '332', '333', '368', '369', '370', '371', '373', '374', '375']

def get_id_from_meta(m):
    return m["path"].split("/")[0]

if __name__ == "__main__":
    """
    splits celebvtext metas into train and validation sets based on distance of the subject arcface embeddings (makes sure that cosine similarity of val samples and their closest training sample is < than threshold)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--divex_file", type=Path, default="data/NERSEMBLE/PROCESSED_FRAMES_EXPR_10K/VALID_DIVEX_METAS.json")
    parser.add_argument("--out_root", type=Path, default="data/NERSEMBLE/PROCESSED_FRAMES_EXPR_10K")
    args = parser.parse_args()

    with open(args.divex_file, "r") as f:
        metas = json.load(f)

    train_metas, val_metas = list(), list()

    for m in metas:
        if get_id_from_meta(m) in NERSEMBLE_VAL_IDS:
            val_metas.append(m)
        else:
            train_metas.append(m)

    train_subjs = np.sort(np.unique(np.array([get_id_from_meta(m) for m in train_metas], dtype=str)))
    val_subjs = np.sort(np.unique(np.array([get_id_from_meta(m) for m in val_metas], dtype=str)))

    args.out_root.mkdir(exist_ok=True, parents=True)
    with open(args.out_root / "TRAIN_METAS.json", "w") as f:
        json.dump(train_metas, f, indent="\t")
    with open(args.out_root / "VAL_METAS.json", "w") as f:
        json.dump(val_metas, f, indent="\t")
    with open(args.out_root / "SUBJECT_SPLIT.json", "w") as f:
        json.dump(dict(train_subjs=train_subjs.tolist(), val_subjs=val_subjs.tolist()),
                  f, indent="\t")

    print(f"Split done. {len(train_metas)} train frames ({len(train_metas) / len(metas)*100:.2f}%); {len(val_metas)} val frames ({len(val_metas) / len(metas)*100:.2f}%).\n\n"
          f"Val ids: {val_subjs}\n\n"
          f"Train ids: {train_subjs}")
