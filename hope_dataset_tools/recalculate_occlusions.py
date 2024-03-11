import os

import cv2
from pathlib import Path

import numpy as np

import shutil
def add_stripe(img:np.ndarray, width):
    top_left = ((img.shape[1] - width)//2, 0)
    bottom_right = ((img.shape[1] + width)//2, img.shape[0])
    cv2.rectangle(img, top_left,bottom_right, (0,0,0), -1)

def __soft_refresh_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    dataset_type = "hope"
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "SynthDynamicOcclusion"

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
        for dir in ["depth", "mask", "mask_visib", "rgb"]:
            img_paths = scene_path / dir
            new_img_paths = new_scene_path / dir
            img_names = os.listdir(img_paths)
            __soft_refresh_dir(new_img_paths)
            for img_name in img_names:
                img_path = img_paths / img_name
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                add_stripe(img, 100)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # cv2.imwrite(str(Path("/home/vojta/PycharmProjects/gtsam_playground/hope_dataset_tools")/"bagr.png"), img)
                cv2.imwrite(str(new_img_paths/img_name), img)


if __name__ == "__main__":
    main()