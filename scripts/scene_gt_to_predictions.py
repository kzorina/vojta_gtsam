# Standard Library
import argparse
import json
import os
import gc
########################
# Add cosypose to my path -> dirty
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import cv2

# Third Party
import numpy as np
import torch
from collections import defaultdict
import pickle


########################
import shutil
from bop_tools import convert_frames_to_bop, export_bop

import cv2

YCBV_OBJECT_NAMES = {"obj_000001": "01_master_chef_can",
    "obj_000002": "02_cracker_box",
    "obj_000003": "03_sugar_box",
    "obj_000004": "04_tomatoe_soup_can",
    "obj_000005": "05_mustard_bottle",
    "obj_000006": "06_tuna_fish_can",
    "obj_000007": "07_pudding_box",
    "obj_000008": "08_gelatin_box",
    "obj_000009": "09_potted_meat_can",
    "obj_000010": "10_banana",
    "obj_000011": "11_pitcher_base",
    "obj_000012": "12_bleach_cleanser",
    "obj_000013": "13_bowl",
    "obj_000014": "14_mug",
    "obj_000015": "15_power_drill",
    "obj_000016": "16_wood_block",
    "obj_000017": "17_scissors",
    "obj_000018": "18_large_marker",
    "obj_000019": "19_large_clamp",
    "obj_000020": "20_extra_large_clamp",
    "obj_000021": "21_foam_brick"}

HOPE_OBJECT_NAMES = {"obj_000001": "AlphabetSoup",
    "obj_000002": "BBQSauce",
    "obj_000003": "Butter",
    "obj_000004": "Cherries",
    "obj_000005": "ChocolatePudding",
    "obj_000006": "Cookies",
    "obj_000007": "Corn",
    "obj_000008": "CreamCheese",
    "obj_000009": "GranolaBars",
    "obj_000010": "GreenBeans",
    "obj_000011": "Ketchup",
    "obj_000012": "MacaroniAndCheese",
    "obj_000013": "Mayo",
    "obj_000014": "Milk",
    "obj_000015": "Mushrooms",
    "obj_000016": "Mustard",
    "obj_000017": "OrangeJuice",
    "obj_000018": "Parmesan",
    "obj_000019": "Peaches",
    "obj_000020": "PeasAndCarrots",
    "obj_000021": "Pineapple",
    "obj_000022": "Popcorn",
    "obj_000023": "Raisins",
    "obj_000024": "SaladDressing",
    "obj_000025": "Spaghetti",
    "obj_000026": "TomatoSauce",
    "obj_000027": "Tuna",
    "obj_000028": "Yogurt"}

OBJECT_NAMES = HOPE_OBJECT_NAMES

def save_preditions_data(output_path, all_predictions):
    with open(output_path, 'wb') as file:
        pickle.dump(all_predictions, file)
    print(f"data saved to {output_path}")

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)


def load_scene_gt(path, label_list=None):
    with open(path) as json_file:
        data: dict = json.load(json_file)
    parsed_data = []
    for k, val in data.items():
        entry = defaultdict(lambda: [])
        for object in val:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            entry[label_list[obj_id - 1]].append(T_cm)
        parsed_data.append(dict(entry))
    return parsed_data

def bogus_px_counts(scene_predictions, shape=(480, 640), size = 0.1):
    ret = []
    for frame in range(len(scene_predictions)):
        entry = {}
        for obj_label in scene_predictions[frame]:
            lst = []
            for obj_inst in range(len(scene_predictions[frame][obj_label])):
                lst.append(int((shape[0]*size) * (shape[1]*size)))
            entry[obj_label] = lst
        ret.append(entry)
    return ret

def main():
    DATASETS_PATH = Path("/home/kzorina/work/bop_datasets/ycbv_test_bop19")
    # DATASET_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]  # hope, synth
    DATASET_NAMES = [
        "000048",  "000050",  "000052",  "000054",  "000056",  "000058",
        "000049",  "000051",  "000053",  "000055",  "000057",  "000059",
    ]
    for DATASET_NAME in DATASET_NAMES[0:]:
        scene_gt_path = DATASETS_PATH/"test"/DATASET_NAME/"scene_gt.json"
        scene_gt = load_scene_gt(scene_gt_path, list(YCBV_OBJECT_NAMES.values()))
        save_preditions_data(DATASETS_PATH/"test"/DATASET_NAME/"frames_prediction.p", scene_gt)
        save_preditions_data(DATASETS_PATH/"test"/DATASET_NAME/"frames_px_counts.p", bogus_px_counts(scene_gt))
        print(f"{DATASETS_PATH/DATASET_NAME}:")


if __name__ == "__main__":
    main()
