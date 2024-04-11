#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-27
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import json
from collections import defaultdict
import pinocchio as pin
import numpy as np


def load_scene_gt(path, label_list=None):
    with open(path) as json_file:
        data: dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = defaultdict(lambda: [])
        frame = i + 1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            entry[label_list[obj_id - 1]].append(T_cm)
        parsed_data.append(entry)
    return parsed_data


def _cam_w2c_pose_from_cam_frame(cam):
    R_w2c = np.array(cam["cam_R_w2c"]).reshape((3, 3))
    t_w2c = np.array(cam["cam_t_w2c"]) / 1000
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_w2c.T
    T_cw[:3, 3] = -R_w2c.T @ t_w2c
    return T_cw



def json_to_pose(element, prefix="cam", suffix="w2c"):
    T_cm = np.zeros((4, 4))
    T_cm[:3, :3] = np.array(element[f"{prefix}_R_{suffix}"]).reshape((3, 3))
    T_cm[:3, :3] = T_cm[:3, :3]
    T_cm[:3, 3] = np.array(element[f"{prefix}_t_{suffix}"]) / 1000
    T_cm[3, 3] = 1
    return T_cm

def compute_t_id_log_err_pairs_for_object(frames, obj_label, camera_poses):
    """Return array whose columns are (frame_id, track_id, log_err)"""
    v = []
    for i in range(len(frames)):
        f = frames[i]
        cam_w2c = pin.SE3(_cam_w2c_pose_from_cam_frame(camera_poses[str(i + 1)]))

        if obj_label in f:
            for pose in f[obj_label]:
                if isinstance(pose, dict):
                    T = cam_w2c * pin.SE3(pose["T_co"])
                    track_id = pose["id"]
                    predicted = pose["valid"]
                    v.append(
                        [
                            i,
                            track_id,
                            np.linalg.norm(pin.log3(T.rotation)),
                            np.linalg.norm(T.translation),
                            predicted,
                            np.linalg.det(pose["Q"]),
                            np.linalg.det(pose["Q"][3:6, 3:6]) ** (1/3),
                            np.linalg.det(pose["Q"][:3, :3]) ** (1/3),
                        ]
                    )
                else:
                    T = cam_w2c * pin.SE3(pose)
                    track_id = 0
                    v.append([i, track_id,  np.linalg.norm(pin.log3(T.rotation)), np.linalg.norm(T.translation)])
    return np.asarray(v)


HOPE_OBJECT_NAMES = {
    "obj_000001": "AlphabetSoup",
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
    "obj_000028": "Yogurt",
}
OBJ_IDS = {
    "AlphabetSoup": 1,
    "BBQSauce": 2,
    "Butter": 3,
    "Cherries": 4,
    "ChocolatePudding": 5,
    "Cookies": 6,
    "Corn": 7,
    "CreamCheese": 8,
    "GranolaBars": 9,
    "GreenBeans": 10,
    "Ketchup": 11,
    "MacaroniAndCheese": 12,
    "Mayo": 13,
    "Milk": 14,
    "Mushrooms": 15,
    "Mustard": 16,
    "OrangeJuice": 17,
    "Parmesan": 18,
    "Peaches": 19,
    "PeasAndCarrots": 20,
    "Pineapple": 21,
    "Popcorn": 22,
    "Raisins": 23,
    "SaladDressing": 24,
    "Spaghetti": 25,
    "TomatoSauce": 26,
    "Tuna": 27,
    "Yogurt": 28,
}


def obj_label_to_id(obj_label: str) -> int:
    return OBJ_IDS[obj_label]
