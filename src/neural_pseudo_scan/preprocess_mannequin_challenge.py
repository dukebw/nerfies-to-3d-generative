import csv
import json
import os
import tarfile
from pprint import pprint

import click
import ffmpeg
import requests
import youtube_dl


def _get_timestamp_bounds_microseconds(txtfile_lines):
    min_microseconds = float("inf")
    max_microseconds = 0
    for l in txtfile_lines[1:]:
        microseconds = float(l[0])
        if microseconds < min_microseconds:
            min_microseconds = microseconds
        if microseconds > max_microseconds:
            max_microseconds = microseconds

    return min_microseconds, max_microseconds


def _read_textfile_lines(txtfile_fpath):
    with open(txtfile_fpath) as f:
        txtfile_reader = csv.reader(f, delimiter=" ")
        txtfile_lines = [l for l in txtfile_reader]

    return txtfile_lines


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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(mannequin_tar)
        os.remove("MannequinChallenge.tar")


@click.command()
@click.option("--video-id")
@click.option("--input-video-path")
def extract_annotated_video_frames(video_id, input_video_path):
    mannequin_train_dir = os.path.join("MannequinChallenge", "train")
    train_video_txtfile_fpaths = os.listdir(mannequin_train_dir)
    for txtfile_fname in train_video_txtfile_fpaths:
        txtfile_lines = _read_textfile_lines(
            os.path.join(mannequin_train_dir, txtfile_fname)
        )

        video_url = txtfile_lines[0][0]
        if video_id not in video_url:
            continue

        min_microseconds, max_microseconds = _get_timestamp_bounds_microseconds(
            txtfile_lines
        )

        min_seconds = min_microseconds / 10 ** 6
        max_seconds = max_microseconds / 10 ** 6

        input_video_path_truncated = os.path.splitext(input_video_path)[0]
        frames_dir = f"{input_video_path_truncated}_{int(min_microseconds)}_{int(max_microseconds)}"
        os.makedirs(frames_dir, exist_ok=True)
        ffmpeg.input(input_video_path).trim(start=min_seconds, end=max_seconds).setpts(
            "PTS-STARTPTS"
        ).output(os.path.join(frames_dir, "%04d.jpg")).run()


@click.command()
def extract_video_info():
    video_infos_all = []
    mannequin_train_dir = os.path.join("MannequinChallenge", "train")
    train_video_txtfile_fpaths = os.listdir(mannequin_train_dir)
    for txtfile_fname in train_video_txtfile_fpaths:
        txtfile_fpath = os.path.join(mannequin_train_dir, txtfile_fname)
        video_url = _read_textfile_lines(txtfile_fpath)[0][0]
        if video_url == "https://www.youtube.com/watch?v=j7xwQ3crVVU":
            print(txtfile_fpath)

        with youtube_dl.YoutubeDL({}) as ydl:
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


@click.command()
@click.option("--video-id")
def get_video_annotation_timestamps(video_id):
    mannequin_train_dir = os.path.join("MannequinChallenge", "train")
    train_video_txtfile_fpaths = os.listdir(mannequin_train_dir)
    for txtfile_fname in train_video_txtfile_fpaths:
        txtfile_lines = _read_textfile_lines(
            os.path.join(mannequin_train_dir, txtfile_fname)
        )

        video_url = txtfile_lines[0][0]
        if video_id not in video_url:
            continue

        min_microseconds, max_microseconds = _get_timestamp_bounds_microseconds(
            txtfile_lines
        )

        min_seconds = int((min_microseconds // 10 ** 6) % 60)
        min_minutes = int(min_microseconds // ((10 ** 6) * 60))
        max_seconds = int((max_microseconds // 10 ** 6) % 60)
        max_minutes = int(max_microseconds // ((10 ** 6) * 60))
        print(f"{min_minutes}:{min_seconds:02d} to {max_minutes}:{max_seconds:02d}")


preprocess_mannequin_challenge.add_command(extract_annotated_video_frames)
preprocess_mannequin_challenge.add_command(download_mannequin_challenge)
preprocess_mannequin_challenge.add_command(extract_video_info)
preprocess_mannequin_challenge.add_command(get_high_res_video_urls)
preprocess_mannequin_challenge.add_command(get_video_annotation_timestamps)


if __name__ == "__main__":
    preprocess_mannequin_challenge()
