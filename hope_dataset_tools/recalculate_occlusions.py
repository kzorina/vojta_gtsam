import os

import cv2
from pathlib import Path
import random
import numpy as np

import shutil
def add_stripe(img:np.ndarray, width):
    top_left = ((img.shape[1] - width)//2, 0)
    bottom_right = ((img.shape[1] + width)//2, img.shape[0])
    cv2.rectangle(img, top_left,bottom_right, (0,0,0), -1)

def add_noise(img, sigma):
    img_type = img.dtype
    img_max = np.iinfo(img_type).max
    rnd_img = np.random.normal(0, img_max * sigma, img.shape)
    img = img.astype(np.float64)
    img = (img + rnd_img)
    img = np.clip(img, 0, img_max)
    img = img.astype(img_type)
    return img
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

def add_smudge(img, length=10):
    psf = np.zeros((50, 50, 3))
    psf = cv2.ellipse(psf,
                      (25, 25),  # center
                      (length, 0),  # axes -- 22 for blur length, 0 for thin PSF
                      15,  # angle of motion in degrees
                      0, 360,  # ful ellipse, not an arc
                      (1, 1, 1),  # white color
                      thickness=-1)  # filled

    psf /= psf[:, :, 0].sum()  # normalize by sum of one channel
    # since channels are processed independently
    imfilt = cv2.filter2D(img, -1, psf)
    return imfilt

def __soft_refresh_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    dataset_type = "hope"
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "SynthDynamicOcclusion"
    DATASET_NAME = "SynthStatic"

    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # scenes = [0]
    for scene_id in scenes:
        print(f"scene_id:{scene_id}")
        scene_name = f"{scene_id:06d}"
        scene_path = DATASETS_PATH / DATASET_NAME / "test_init" / scene_name
        new_scene_path = DATASETS_PATH / DATASET_NAME / "test" / scene_name
        __soft_refresh_dir(new_scene_path)
        shutil.copyfile(scene_path/"scene_camera.json", new_scene_path/"scene_camera.json")
        shutil.copyfile(scene_path/"scene_gt.json", new_scene_path/"scene_gt.json")
        for dir in ["rgb", "depth", "mask", "mask_visib"]:
            img_paths = scene_path / dir
            new_img_paths = new_scene_path / dir
            img_names = os.listdir(img_paths)
            __soft_refresh_dir(new_img_paths)
            for img_name in img_names:
                img_path = img_paths / img_name
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

                if dir == "rgb":
                    if random.uniform(0, 1) < 0.4:
                        img = add_smudge(img, 5)
                    # img = add_noise(img, 0.08)
                # add_stripe(img, 100)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # cv2.imwrite(str(Path("/home/vojta/PycharmProjects/gtsam_playground/hope_dataset_tools")/"bagr.png"), img)
                cv2.imwrite(str(new_img_paths/img_name), img)


if __name__ == "__main__":
    # add_noise(np.zeros((480, 640)), 10)
    main()