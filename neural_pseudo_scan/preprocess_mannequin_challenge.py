import csv
import json
import os
import tarfile
from pprint import pprint

import click
import requests
import youtube_dl


@click.group()
def preprocess_mannequin_challenge():
    pass


@click.command()
def download_mannequin_challenge():
    if not os.path.exists("MannequinChallenge"):
        response = requests.get(
            "https://storage.googleapis.com/mannequinchallenge-data/MannequinChallenge.tar"
        )
        with open("MannequinChallenge.tar", "wb") as f:
            f.write(response.content)
        with tarfile.open("MannequinChallenge.tar") as mannequin_tar:
            mannequin_tar.extractall()
        os.remove("MannequinChallenge.tar")


@click.command()
def extract_video_info():
    video_infos_all = []
    ydl_opts = {}
    mannequin_train_dir = os.path.join("MannequinChallenge", "train")
    train_video_txtfile_fpaths = os.listdir(mannequin_train_dir)
    for txtfile_fname in train_video_txtfile_fpaths:
        txtfile_fpath = os.path.join(mannequin_train_dir, txtfile_fname)
        with open(txtfile_fpath) as f:
            txtfile_reader = csv.reader(f)
            video_url = [l for l in txtfile_reader][0][0]
        if video_url == "https://www.youtube.com/watch?v=j7xwQ3crVVU":
            print(txtfile_fpath)

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                video_info = ydl.extract_info(video_url, download=False)
            except youtube_dl.utils.DownloadError:
                continue
        video_infos_all.append(video_info)
    with open("mannequin-challenge-video-info.json", "w") as f:
        json.dump(video_infos_all, f, indent=8)


@click.command()
def get_high_res_video_urls():
    video_infos_high_res = []
    with open("mannequin-challenge-video-info.json", "r") as f:
        video_infos_all = json.load(f)

    for video_info in video_infos_all:
        video_max_dim = 0
        for fmt in video_info["formats"]:
            if (fmt["width"] is not None) and (fmt["width"] > video_max_dim):
                video_max_dim = fmt["width"]
            if (fmt["height"] is not None) and (fmt["height"] > video_max_dim):
                video_max_dim = fmt["height"]

        if video_max_dim > 1920:
            video_infos_high_res.append(video_info)

    with open("video-infos-high-res.json", "w") as f:
        json.dump(video_infos_high_res, f)


preprocess_mannequin_challenge.add_command(download_mannequin_challenge)
preprocess_mannequin_challenge.add_command(extract_video_info)
preprocess_mannequin_challenge.add_command(get_high_res_video_urls)


if __name__ == "__main__":
    preprocess_mannequin_challenge()
