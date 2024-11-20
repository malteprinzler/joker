import copy
import glob
import warnings
import re
import matplotlib.pyplot as plt
import torch
import tqdm
from torchvision.io import read_image, ImageReadMode
import json
import numpy as np
import random
from pathlib import Path
from joker.utils import *
from collections import defaultdict
from transformers import CLIPTokenizer


class JokerDataset(torch.utils.data.Dataset):
    PERSON_SYNONYMS = ["person", "persons", "people", "woman", "women", "man", "men", "child", "children", "boy",
                       "boys", "girl", "girls", "baby", "babies", "actress", "actor", "priest", ]

    def __init__(self,
                 root,
                 split,
                 meta_fp,
                 nsamples=-1,
                 deterministic=False,
                 normal_dir="bfm_normals-512",
                 caption_dir="blip_captions",
                 img_dir="images-512",
                 img_stem="cam_00",
                 tokenizer=None,
                 sort_by_divex=True,
                 deterministic_shuffle=False,
                 uncondition_prob=0.,
                 use_for_avg_quant=True,
                 ):
        """

        :param root: str,
        :param split: str, one of ["val", "train"]
        :param meta_fp: str, path to dataset file metas
        :param nsamples: int, nr of dataset samples (-1 for using all samples)
        :param deterministic: bool, whether to use seeded sampling for randomly sampling reference and target images
        :param normal_dir: str, name of normal map dir
        :param caption_dir: str, name of caption map dir
        :param img_stem: str, stem of image files
        :param tokenizer: transformers.CLIPTokenizer, optional
        :param sort_by_divex: bool, whether to sort samples by divex (expression amplitude and intra-video diversity)
                            or not
        :param deterministic_shuffle: bool, whether to deterministically shuffle samples
        """

        assert split in ["train", "val", "all"]

        self.root = Path(root)
        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
            )
        self.nref = 1
        self.tokenizer = tokenizer
        self.split = split
        self.nsamples = nsamples
        self.deterministic = deterministic
        self.img_dir = img_dir
        self.normal_dir = normal_dir
        self.caption_dir = caption_dir
        self.img_stem = img_stem
        self.sort_by_divex = sort_by_divex
        self.deterministic_shuffle = deterministic_shuffle
        self.uncondition_prob = uncondition_prob
        self.use_for_avg_quant = use_for_avg_quant
        if self.split in ["val", "all"]:
            self.deterministic = True
            assert uncondition_prob == 0

        self.sample_list = self.get_sample_list(meta_fp)

    def get_sample_list(self, meta_fp):
        """
        obtaining list of dataset samples to iterate over, sorted by divex score
        (i.e. high expressiveness and intra-scene expression diversity)

        :param meta_fp:
        :return:
        """
        assert self.split in str(meta_fp).lower()
        with open(meta_fp, "r") as f:
            metas = json.load(f)

        if self.sort_by_divex:
            divex_scores = np.array([sample["divex"] for sample in metas])
            idcs = np.argsort(divex_scores)[::-1]
            metas = [metas[idx] for idx in idcs]

        if self.deterministic_shuffle:
            assert not self.sort_by_divex
            random.Random(0).shuffle(metas)

        if self.nsamples > 0:
            metas = metas[:self.nsamples]

        return metas

    def check_captions(self):
        """
        utility function for checking captions (checking if all captions contain one of self.PERSON_SYNONYMS)
        :return:
        """
        for idx in tqdm.tqdm(np.random.permutation(len(self)), desc="Checking captions"):
            sample = self.sample_list[idx]
            frame_path = (self.root / sample["path"]).parents[1]
            caption_path = list((frame_path / self.caption_dir).iterdir())[0]
            caption = self.load_caption(caption_path)
            if find_words_in_str(caption, self.PERSON_SYNONYMS) is None:
                print(f"ERROR: Couldnt find key word in caption {caption} from {caption_path}, idx: {idx}")

    def get_sample_file_paths(self, idx):
        sample = self.sample_list[idx]
        target_image_path = self.root / sample["path"]
        frame_path = target_image_path.parents[1]

        mask_dir = frame_path / "mask-512"
        normal_dir = frame_path / self.normal_dir
        caption_path = frame_path / self.caption_dir / f"{self.img_stem}.txt"

        if self.deterministic:
            rg = random.Random(idx)  # deterministic reference samples for validation set
        else:
            rg = random

        if "farest_neighbors" in sample:
            # sampling ref images from most distant frames of same sequence
            ref_candidates = [self.root / p for p in sample["farest_neighbors"]]
            ref_candidates = ref_candidates[:3]
            assert self.nref == 1
            ref_indices = rg.choices(list(range(len(ref_candidates))),
                                     weights=2 ** np.arange(len(ref_candidates))[::-1],
                                     k=1)
            ref_image_paths = [ref_candidates[i] for i in ref_indices]
        else:
            ref_frame_candidates = [p for p in frame_path.parent.iterdir() if p != frame_path]
            ref_frame = rg.choice(ref_frame_candidates)
            assert self.nref == 1
            ref_image_paths = [ref_frame / self.img_dir / f"{self.img_stem}.jpg"]

        target_mask_path = mask_dir / target_image_path.name.replace(".jpg", ".png")
        target_normal_path = normal_dir / target_image_path.name

        return dict(
            caption_path=caption_path,
            target_image_path=target_image_path,
            target_mask_path=target_mask_path,
            target_normal_path=target_normal_path,
            ref_image_paths=ref_image_paths,
        )

    @torch.no_grad()
    def __getitem__(self, idx):
        sample_paths_dict = self.get_sample_file_paths(idx)

        caption = load_caption(sample_paths_dict["caption_path"])
        image = read_image(str(sample_paths_dict["target_image_path"]), mode=ImageReadMode.RGB)
        if sample_paths_dict["target_mask_path"].exists():
            mask = read_image(str(sample_paths_dict["target_mask_path"]), mode=ImageReadMode.GRAY)
        else:
            assert self.split != "val"  # assuming masks are available for validation samples
            # warnings.warn(f"Didnt find mask file {sample_paths_dict['target_mask_path']}")
            mask = torch.ones_like(image[:1]) * 255
        normal = read_image(str(sample_paths_dict["target_normal_path"]), mode=ImageReadMode.RGB)
        ref_images = [read_image(str(f), mode=ImageReadMode.RGB) for f in sample_paths_dict["ref_image_paths"]]

        # normalizing data range from 0...255 -> 0 ... 1, except for normal maps: -1 ... 1
        image = image.float() / 255.0  # watch out, all images are scaled from 0 ... 1 now, including normal img and ref_img
        mask = mask.float() / 255.0
        normal = normal.float() / 127.5 - 1.
        ref_images = torch.stack([img.float() / 255.0 for img in ref_images])

        # tokenizing caption
        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
        caption_ids = tokenize_caption(caption,
                                       tokenizer=self.tokenizer)[0]

        return {
            "img": image,  # torch.Tensor(float32), 3 x H x W, 0 ... 1
            "mask": mask,  # torch.Tensor(float32), 1 x H x W, 0 ... 1
            "normal": normal,  # torch.Tensor(float32), C x H x W, -1 ... 1
            "ref_imgs": ref_images,  # torch.Tensor(float32), Nref x 3 x H x W, 0 ... 1
            "caption_ids": caption_ids,  # torch.Tensor(int64), 77,
            "caption": caption,  # str
        }

    def visualize_sample(self, i):
        """
        visualizing sample with index i
        :param i: int, sample index
        :return:
        """
        sample = self.__getitem__(i)
        img = sample["img"]
        ref_imgs = sample["ref_imgs"]
        normal = sample["normal"]
        caption = sample["caption"]
        mask = sample["mask"]
        nref = len(ref_imgs)

        fig, axes = plt.subplots(ncols=nref + 3, nrows=1, figsize=(6 * (nref + 3), 6 * 2))
        axes[0].imshow(img.permute(1, 2, 0).cpu())
        axes[1].imshow(mask[0])
        axes[2].imshow(normal.permute(1, 2, 0).cpu() * .5 + .5)
        for j in range(nref):
            axes[3 + j].imshow(ref_imgs[j].permute(1, 2, 0).cpu())
        plt.suptitle(caption)
        plt.show()

    def __len__(self):
        return len(self.sample_list)


class NeRSembleDataset(JokerDataset):
    """
    similar to JokerDataset but assumes that elements of self.sample_list are paths to directories of images with
    multi-view data. Will randomly sample one of the views
    """

    def get_balanced_samples_from_divex_metas(self, divex_metas):
        # ensures all subjects are sampled equally
        subject_metas = defaultdict(list)
        for meta in divex_metas:
            subject = meta["path"].split("/")[0]
            subject_metas[subject].append(meta)

        subjects = sorted(subject_metas.keys())
        balanced_frames = list()
        i = 0
        while len(balanced_frames) < len(divex_metas):
            subject = subjects[i % len(subjects)]
            if len(subject_metas[subject]) > 0:
                img_path = Path(subject_metas[subject].pop(0)["path"])
                balanced_frames.append(str(img_path.parents[1]))
            i += 1
        return balanced_frames

    def get_unbalanced_samples_from_divex_metas(self, divex_metas):
        # just goes for highest divex-score
        divex_scores = np.array([meta["divex"] for meta in divex_metas])
        idcs = np.argsort(divex_scores)[::-1]
        unbalanced_frames = list()
        for idx in idcs:
            meta = divex_metas[idx]
            img_path = Path(meta["path"])
            unbalanced_frames.append(str(img_path.parents[1]))
        return unbalanced_frames

    def get_sample_list(self, meta_fp):
        assert self.split in str(meta_fp).lower()
        with open(meta_fp, "r") as f:
            metas = json.load(f)

        assert self.sort_by_divex, NotImplementedError(
            "NeRSemble Dataset only implemented for sorting samples by divex")
        if self.split == "train":
            nersemble_samples = self.get_balanced_samples_from_divex_metas(
                metas)  # ensure that during training as many subjects are seen as possible
        else:
            nersemble_samples = self.get_unbalanced_samples_from_divex_metas(metas)
        sample_list = [self.root / f for f in nersemble_samples]

        if self.nsamples > 0:
            sample_list = sample_list[:self.nsamples]

        if self.deterministic_shuffle:
            random.Random(0).shuffle(metas)

        return sample_list

    def get_sample_file_paths(self, idx):
        sample = self.sample_list[idx]
        frame_path = sample
        img_dir = frame_path / self.img_dir
        mask_dir = frame_path / "mask-512"
        normal_dir = frame_path / self.normal_dir
        caption_path = frame_path / self.caption_dir / "cam_222200037.txt"
        img_paths = sorted([p for p in img_dir.iterdir() if p.name.endswith(".jpg")])

        if self.deterministic:
            rg = random.Random(idx)  # deterministic reference samples for validation set
        else:
            rg = random

        target_image_idx = rg.randint(0, len(img_paths) - 1)
        target_image_path = img_paths.pop(target_image_idx)
        ref_frame_candidates = [s for s in self.sample_list if s != sample and sample.parents[2] == s.parents[2]]

        if len(ref_frame_candidates) == 0:  # falling back to target frame in case no suitable ref frame is available
            ref_frame_candidates = [frame_path]
        ref_image_paths = list()
        for i in range(self.nref):
            ref_frame_idx = rg.randint(0, len(ref_frame_candidates) - 1)
            ref_frame = ref_frame_candidates[ref_frame_idx]
            ref_img_dir = ref_frame / self.img_dir
            ref_image_candidate_paths = sorted([p for p in ref_img_dir.iterdir() if p.name.endswith(".jpg")])
            if len(ref_image_candidate_paths) > 1:
                ref_image_candidate_paths = [p for p in ref_image_candidate_paths if
                                             p.name != target_image_path.name]
            ref_image_paths.append(rg.choice(ref_image_candidate_paths))

        target_mask_path = mask_dir / target_image_path.name.replace(".jpg", ".png")
        target_normal_path = normal_dir / target_image_path.name

        return dict(
            caption_path=caption_path,
            target_image_path=target_image_path,
            target_mask_path=target_mask_path,
            target_normal_path=target_normal_path,
            ref_image_paths=ref_image_paths,
        )

    def check_captions(self):
        for idx in tqdm.tqdm(range(0, len(self)), desc="Checking captions"):
            sample = self.sample_list[idx]
            frame_path = sample
            caption_path = list((frame_path / self.caption_dir).iterdir())[0]
            caption = self.load_caption(caption_path)
            if find_words_in_str(caption, self.PERSON_SYNONYMS) is None:
                print(f"ERROR: Couldnt find key word in caption {caption} from {caption_path}, idx: {idx}")


class FolderDataset(JokerDataset):
    """
    simple dataset that loads all samples from a single folder.
    assuming folder with format
        - 000000_ctrl.jpg
        - 000000_drive.jpg
        - 000000_prompt.txt
        - 000000_ref.jpg
        - 000001_...

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, meta_fp=None, split="all")

    def get_sample_list(self, meta_fp=None):
        sample_list = [Path(p) for p in sorted(glob.glob(str(self.root / "*_ref.*")))]
        return sample_list

    def get_sample_file_paths(self, idx):
        ref_path = self.sample_list[idx]
        stem = str(ref_path.parent / ref_path.name.split("_")[0])
        caption_path = Path(stem + "_prompt.txt")
        target_image_path = Path(stem + "_drive.jpg")
        target_mask_path = Path(stem + "_mask.png")
        target_normal_path = Path(stem + "_ctrl.jpg")
        ref_image_paths = [Path(stem + "_ref.jpg")]
        return dict(
            caption_path=caption_path,
            target_image_path=target_image_path,
            target_mask_path=target_mask_path,
            target_normal_path=target_normal_path,
            ref_image_paths=ref_image_paths,
        )


class RandomMixedDataset(torch.utils.data.Dataset):
    def __init__(self, ds_targets: list, ds_kwargs: list, weights=None, **kwargs):
        """
        mixes two datasets weighted by weights. __getitem__ randomly samples from datasets; don't use for validation
        """
        self.datasets = list()

        for ds_target, ds_kwarg in zip(ds_targets, ds_kwargs):
            kwargs_ = copy.deepcopy(kwargs)
            kwargs_.update(ds_kwarg)
            self.datasets.append(import_obj(ds_target)(**kwargs_))
        self.ndatasets = len(self.datasets)
        self.weights = np.array(weights) if weights is not None else np.array([1.] * self.ndatasets)
        self.weights = self.weights.astype(float) / np.sum(self.weights)
        self.subset_lengths = np.array([len(ds) for ds in self.datasets])

    def __len__(self):
        return sum(self.subset_lengths)

    def __getitem__(self, item):
        subset_idx = np.random.choice(self.ndatasets, p=self.weights)
        sample_idx = np.random.choice(self.subset_lengths[subset_idx])
        return self.datasets[subset_idx][sample_idx]

    def visualize_sample(self, i):
        subset_idx = np.random.choice(self.ndatasets, p=self.weights)
        sample_idx = np.random.choice(self.subset_lengths[subset_idx])
        self.datasets[subset_idx].visualize_sample(sample_idx)


def tokenize_caption(caption, tokenizer):
    clean_input_ids = tokenizer.encode(caption)

    max_len = tokenizer.model_max_length

    if len(clean_input_ids) > max_len:
        clean_input_ids = clean_input_ids[:max_len]
    else:
        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
        )

    clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
    clean_input_ids = clean_input_ids.unsqueeze(0)

    return clean_input_ids


def find_words_in_str(s, words):
    """
    finds subject word in s and returns which word was found, as well as start and end idcs of word in s
    """
    for w in words:
        match = re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)
        if match is not None:
            return w, match.start(), match.end()

    return None


def load_caption(p):
    try:
        with open(p, "r") as f:
            caption = f.readline().strip().lower()
    except UnicodeDecodeError as e:
        warnings.warn(f"Couldnt load {p}: {e}\nReturning empty caption.")
        caption = ""
    return caption


if __name__ == "__main__":

    # # CelebVText Dataset
    # ds = JokerDataset(
    #     root="",
    #     split="train",
    #     meta_fp="",
    #     deterministic=True,
    #     nsamples=50,
    # )

    # # Nersemble Dataset
    # ds = NeRSembleDataset(
    #     root="",
    #     split="train",
    #     meta_fp="",
    #     deterministic=True,
    #     nsamples=50,
    # )

    # Own Dataset
    ds = JokerDataset(
        root="",
        split="all",
        meta_fp="",
        deterministic=True,
        nsamples=-1,
        sort_by_divex=False,
        deterministic_shuffle=True,
        img_stem="cam_000"
    )

    print(len(ds))
    # ds.check_captions()
    for i in tqdm.tqdm(np.random.RandomState(0).permutation(len(ds))):
        # ds[i]
        print(i)
        ds.visualize_sample(i)
