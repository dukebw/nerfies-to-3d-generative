import os
from glob import glob

import click
import cv2
import face_alignment
import numpy as np
import skimage.io as skio
import torch
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from skimage.transform import estimate_transform, warp
from tqdm import tqdm


@click.group()
def run_deca_on_video():
    pass


@click.command()
@click.option("--downsample-factor", type=float, default=1.0)
@click.option("--input-images-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
def extract_depth(downsample_factor, input_images_dir, output_dir):
    deca_cfg.model.use_tex = True
    crop_size = 224
    scale = 1.25

    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for extension in ["jpg", "png"]:
        image_paths += glob(os.path.join(input_images_dir, f"*.{extension}"))
    image_paths = sorted(image_paths)

    face_detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False
    )

    deca = DECA(config=deca_cfg)

    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]
        image = skio.imread(img_path)
        image = np.array(image)
        image = cv2.resize(image, (0, 0), fx=downsample_factor, fy=downsample_factor)

        landmarks_all_faces = face_detector.get_landmarks_from_image(image)
        if landmarks_all_faces is None:
            continue

        image = image / 255.0

        for face_idx, landmarks in enumerate(landmarks_all_faces):
            keypoints = landmarks.squeeze()

            left = np.min(keypoints[:, 0])
            right = np.max(keypoints[:, 0])
            top = np.min(keypoints[:, 1])
            bottom = np.max(keypoints[:, 1])

            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
            )
            size = int(old_size * scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )

            dst_pts = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
            similarity_transform = estimate_transform("similarity", src_pts, dst_pts)

            image_warped = warp(
                image, similarity_transform.inverse, output_shape=(crop_size, crop_size)
            )
            image_warped = image_warped.transpose(2, 0, 1)
            image_warped = torch.tensor(image_warped).float()
            image_warped = image_warped.cuda()[None, ...]

            codedict = deca.encode(image_warped)
            opdict, visdict = deca.decode(codedict)

            depth_image = deca.render.render_depth(
                opdict["transformed_vertices"]
            ).repeat(1, 3, 1, 1)
            visdict["depth_images"] = depth_image
            depth_image = util.tensor2image(depth_image.squeeze(0))

            # NOTE(brendan): depth_image is HxWx3 in [0, 1]
            depth_image = warp(
                depth_image, similarity_transform, output_shape=image.shape
            )
            cv2.imwrite(
                os.path.join(output_dir, f"{img_name}_{face_idx}_depth.jpg"),
                (255.0 * depth_image).astype(np.uint8),
            )
            depth_alpha = 0.9 * (depth_image > 0).astype(np.float32)

            img_vis_depth = ((1 - depth_alpha) * image[:, :, ::-1]) + (
                depth_alpha * depth_image
            )
            cv2.imwrite(
                os.path.join(output_dir, f"{img_name}_{face_idx}_depth_vis.jpg"),
                (255.0 * img_vis_depth).astype(np.uint8),
            )


run_deca_on_video.add_command(extract_depth)


if __name__ == "__main__":
    run_deca_on_video()
