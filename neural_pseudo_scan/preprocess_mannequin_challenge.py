import csv
import json
import os
from pprint import pprint
import tarfile

import requests
import youtube_dl


def preprocess_mannequin_challenge():
    if not os.path.exists("MannequinChallenge"):
        response = requests.get(
            "https://storage.googleapis.com/mannequinchallenge-data/MannequinChallenge.tar"
        )
        with open("MannequinChallenge.tar", "wb") as f:
            f.write(response.content)
        with tarfile.open("MannequinChallenge.tar") as mannequin_tar:
            mannequin_tar.extractall()
        os.remove("MannequinChallenge.tar")

    high_res_video_urls = []
    ydl_opts = {}
    mannequin_train_dir = os.path.join("MannequinChallenge", "train")
    train_video_txtfile_fpaths = os.listdir(mannequin_train_dir)
    for txtfile_fname in train_video_txtfile_fpaths:
        txtfile_fpath = os.path.join(mannequin_train_dir, txtfile_fname)
        with open(txtfile_fpath) as f:
            txtfile_reader = csv.reader(f)
            video_url = [l for l in txtfile_reader][0][0]
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                video_info = ydl.extract_info(video_url, download=False)
            except youtube_dl.utils.DownloadError:
                continue

        video_max_dim = 0
        for fmt in video_info["formats"]:
            if (fmt["width"] is not None) and (fmt["width"] > video_max_dim):
                video_max_dim = fmt["width"]
                video_max_dim_format = fmt
            if (fmt["height"] is not None) and (fmt["height"] > video_max_dim):
                video_max_dim = fmt["height"]
                video_max_dim_format = fmt
        if video_max_dim > 1920:
            high_res_video_urls.append(video_url)
        print(high_res_video_urls)
    with open("high-res-video-urls.json", "w") as f:
        json.dump(high_res_video_urls, f)


if __name__ == "__main__":
    preprocess_mannequin_challenge()
