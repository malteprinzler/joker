"""
CelebVText Downloader
"""
import argparse
import os
import json
import cv2
import multiprocessing
from multiprocessing import Pool
from functools import partial
import tqdm
import numpy as np


def download(video_path, ytb_id, proxy=None):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, defalut None
    """
    if proxy is not None:
        proxy_cmd = "--proxy {}".format(proxy)
    else:
        proxy_cmd = ""
    if not os.path.exists(video_path):
        down_video = " ".join([
            "yt-dlp",
            proxy_cmd,
            '-f', "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
            '--skip-unavailable-fragments',
            '--merge-output-format', 'mp4',
            "https://www.youtube.com/watch?v=" + ytb_id, "--output",
            video_path, "--external-downloader", "aria2c",
            "--external-downloader-args", '"-x 16 -k 1M"'
        ])
        print(down_video)
        status = os.system(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")


def process_ffmpeg(raw_vid_path, save_folder, save_vid_name, bbox, time):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    time: begin_sec, end_sec
    """

    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)

        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, leftx, right = bbox
        h = bottom - top
        w = right - leftx
        c = min(h, w) // 2
        c_h = (top + bottom) / 2
        c_w = (leftx + right) / 2

        top, bottom = c_h - c, c_h + c
        leftx, right = c_w - c, c_w + c
        return top, bottom, leftx, right

    def denorm(bbox, height, width):
        top = round(bbox[0] * height)
        bottom = round(bbox[1] * height)
        left = round(bbox[2] * width)
        right = round(bbox[3] * width)
        return top, bottom, left, right

    out_path = os.path.join(save_folder, save_vid_name)
    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(denorm(expand(bbox, 0.02), height, width))
    start_sec, end_sec = time

    cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={right - left}:h={bottom - top}:x={left}:y={top} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error -y {out_path}"
    os.system(cmd)
    return out_path


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    samples = list()
    for key, val in data_dict.items():
        save_name = key + ".mp4"
        ytb_id = val['ytb_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']
        bbox = [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]
        samples.append([ytb_id, save_name, time, bbox])
    return samples


def process_sample(sample, processed_vid_root, tmp_raw_vid_root, tmp_processed_vid_root, proxy=None):
    vid_id, save_vid_name, time, bbox = sample
    tmp_raw_vid_path = os.path.join(tmp_raw_vid_root, vid_id + ".mp4")

    # Downloading is io bounded and processing is cpu bounded.
    # It is better to download all videos firstly and then process them via multiple cpu cores.
    try:
        download(tmp_raw_vid_path, vid_id, proxy)
        tmp_processed_vid_path = process_ffmpeg(tmp_raw_vid_path, tmp_processed_vid_root, save_vid_name, bbox, time)
        os.system(f"mv {tmp_processed_vid_path} {processed_vid_root}")
        os.system(f"rm {tmp_raw_vid_path}")
    except Exception as e:
        print(f"ERROR: {vid_id}; {e}")


if __name__ == '__main__':
    """
    Downloads celebvtext dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_file", default='data/CELEBV_TEXT/celebvtext_info.json', type=str)
    parser.add_argument("--out_dir", default='data/CELEBV_TEXT/VIDEOS', type=str)
    parser.add_argument("--tmp_dir_raw", default='/tmp/CELEBV_TEXT/RAW', type=str)
    parser.add_argument("--tmp_dir_processed", default='/tmp/CELEBV_TEXT/PROCESSED', type=str)
    parser.add_argument("--nworkers", default=10, type=int)
    args = parser.parse_args()

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))
    proxy = None  # proxy url example, set to None if not use

    # os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir_raw, exist_ok=True)
    os.makedirs(args.tmp_dir_processed, exist_ok=True)

    data = load_data(args.url_file)

    my_idcs = np.array_split(np.arange(len(data)), njobs)[jobid]
    data = [data[i] for i in my_idcs]

    worker_fn = partial(process_sample,
                        processed_vid_root=args.out_dir,
                        tmp_raw_vid_root=args.tmp_dir_raw,
                        tmp_processed_vid_root=args.tmp_dir_processed,
                        proxy=proxy)

    # for sample in tqdm.tqdm(data, "Video Files"):
    #     worker_fn(sample)

    multiprocessing.set_start_method("spawn")
    with Pool(args.nworkers) as p:
        list(tqdm.tqdm(p.imap_unordered(worker_fn, data), total=len(data), desc="Processing Videos"))
