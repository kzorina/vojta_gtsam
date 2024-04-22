from pathlib import Path

from gtsam.symbol_shorthand import B, V, X, L
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

def plot_cov_evolution(axis1, axis2,  track_id, frames_refined, linewidth=1, color='b'):
    Qs_t = np.full((len(frames_refined)), np.nan)
    Qs_r = np.full((len(frames_refined)), np.nan)
    for frame in range(len(frames_refined)):
        if frame == 150:
            print('')
        for obj_label in frames_refined[frame]:
                for i in range(len(frames_refined[frame][obj_label])):
                    obj_id = frames_refined[frame][obj_label][i]['id']
                    if obj_id > 10 ** 8:
                        obj_id = int((obj_id - L(0)) * 10 ** (-6))
                    if obj_id == track_id:
                        Q = frames_refined[frame][obj_label][i]['Q']
                        Qs_r[frame] = np.linalg.det(Q[:3, :3]) ** 0.5
                        Qs_t[frame] = np.linalg.det(Q[3:6, 3:6]) ** 0.5
                        break

    print("")

    axis1.plot(np.arange(len(Qs_r)), Qs_r, "-", label=f"Qs_r", linewidth=linewidth, color=color)
    axis1.hlines(y=0.001, xmin=0, xmax=len(Qs_r), linewidth=1, color='black')
    axis2.plot(np.arange(len(Qs_t)), Qs_t, "-", label=f"Qs_t", linewidth=linewidth, color=color)
    axis2.hlines(y=0.000025, xmin=0, xmax=len(Qs_t), linewidth=1, color='black')
    axis1.set_ylim([0.0, 10**(-1)])
    axis2.set_ylim([0.0, 10**(-1)])
    axis1.set_xlabel("frame")
    axis1.set_ylabel("det(cov R)^0.5")
    #
    axis2.set_xlabel("frame")
    axis2.set_ylabel("det(cov t)^0.5")

def main():
    ds_path = DATASET_ROOT / "SynthDynamicOcclusion" / "test" / "000001"
    frames_refined_prediction = pickle.load(open(ds_path / "frames_refined_prediction.p", "rb"))
    frames_refined_prediction_new = pickle.load(open(ds_path / "frames_refined_prediction_new.p", "rb"))
    frames_refined_prediction_swap = pickle.load(open(ds_path / "frames_refined_prediction_swap.p", "rb"))
    frames_prediction = pickle.load(open(ds_path / "frames_prediction.p", "rb"))
    scene_gt = load_scene_gt(ds_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
    camera_poses = load_scene_camera(ds_path / "scene_camera.json")

    fig, ax = plt.subplots(2)
    plt.grid(axis='y')

    all_object_labels = set()
    for frame in scene_gt:
        all_object_labels.update(frame.keys())
    all_object_labels = sorted(all_object_labels)
    # obj_label = all_object_labels[3]
    track_id = 1

    plot_cov_evolution(ax[0], ax[1], track_id, frames_refined_prediction_swap, linewidth=7, color='b')
    plot_cov_evolution(ax[0], ax[1], track_id, frames_refined_prediction_new, linewidth=4, color='r')
    plot_cov_evolution(ax[0], ax[1], track_id, frames_refined_prediction, linewidth=1, color='g')
    plt.show()

if __name__ == "__main__":
    main()