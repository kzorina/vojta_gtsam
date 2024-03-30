from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from utils import (
    load_scene_gt,
    HOPE_OBJECT_NAMES,
    OBJ_IDS,
    compute_t_id_log_err_pairs_for_object,
)
import json
import pinocchio as pin
import copy

DATASET_ROOT = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")


def load_scene_camera(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        entry["cam_K"] = np.array(data[str(i+1)]["cam_K"]).reshape((3, 3))
        T_cw = np.zeros((4, 4))
        T_cw[:3, :3] = np.array(data[str(i+1)]["cam_R_w2c"]).reshape((3, 3))
        T_cw[:3, 3] = np.array(data[str(i+1)]["cam_t_w2c"])/1000
        T_cw[3, 3] = 1
        entry["T_cw"] = T_cw
        parsed_data.append(entry)
    return parsed_data

def derivative(y, dt=1/30):
    dt_y = copy.deepcopy(y)
    dt_y[0] = pin.SE3.Identity()
    for i in range(len(y) - 1):
        T_wo_1 = y[i]
        T_wo_2 = y[i + 1]
        T12:pin.SE3 = T_wo_2 * T_wo_1.inverse()
        dt_w = pin.log3(T12.rotation) / dt
        dt_t = T12.translation / dt
        dt_T_wo = pin.SE3(pin.exp3(dt_w), dt_t)
        dt_y[i] = dt_T_wo
    return dt_y

def split_d(d):
    t = np.zeros((len(d)))
    r = np.zeros((len(d)))
    for i in range(len(d)):
        T:pin.SE3 = d[i]
        t[i] = np.linalg.norm(T.translation)
        # t[i] = T.translation[1]
        r[i] = np.linalg.norm(pin.log3(T.rotation))
    return t, r


def main():
    ds_path = DATASET_ROOT / "SynthDynamicOcclusion" / "test" / "000000"
    frames_refined_prediction = pickle.load(open(ds_path / "frames_refined_prediction.p", "rb"))
    frames_prediction = pickle.load(open(ds_path / "frames_prediction.p", "rb"))
    scene_gt = load_scene_gt(ds_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
    camera_poses = load_scene_camera(ds_path / "scene_camera.json")


    all_object_labels = set()
    for frame in scene_gt:
        all_object_labels.update(frame.keys())
    all_object_labels = sorted(all_object_labels)

    object_label = all_object_labels[0]
    d0 = []
    for frame in range(len(scene_gt)):
        T_co = pin.SE3(scene_gt[frame][object_label][0])
        T_cw = pin.SE3(camera_poses[frame]['T_cw'])
        T_wo = T_cw.inverse() * T_co
        d0.append(T_wo)
    d1 = derivative(d0)
    d2 = derivative(d1)
    print("")

    fig, ax = plt.subplots(3, 2)
    plt.grid(axis='y')

    t,r = split_d(d0)
    ax[0, 0].plot(np.arange(len(t)), t, "-", label=f"d0_t")
    ax[0, 0].yaxis.grid(True, which='major', linestyle='--')
    ax[0, 1].plot(np.arange(len(r)), r, "-", label=f"d0_r")
    ax[0, 0].set_title("d0_t")
    ax[0, 1].set_title("d0_r")

    t, r = split_d(d1)
    ax[1, 0].plot(np.arange(len(t)), t, "-", label=f"d1_t")
    ax[1, 0].yaxis.grid(True, which='major', linestyle='--')
    ax[1, 1].plot(np.arange(len(r)), r, "-", label=f"d1_r")
    ax[1, 0].set_title("d1_t")
    ax[1, 1].set_title("d1_r")

    t, r = split_d(d2)
    ax[2, 0].plot(np.arange(len(t)), t, "-", label=f"d2_t")
    ax[2, 0].yaxis.grid(True, which='major', linestyle='--')
    ax[2, 0].set_ylim(-1, 1)
    ax[2, 1].plot(np.arange(len(r)), r, "-", label=f"d2_r")
    ax[2, 0].set_title("d2_t")
    ax[2, 1].set_title("d2_r")

    # ax[0].set_xlabel("frame gap size")
    # ax[0].set_ylabel("Recall")
    #
    # ax[1].set_xlabel("frame gap size")
    # ax[1].set_ylabel("Precision")
    #
    # ax[0].legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.25),
    #     ncol=4,
    #     fancybox=True,
    #     shadow=True,
    # )
    plt.show()

if __name__ == "__main__":
    main()