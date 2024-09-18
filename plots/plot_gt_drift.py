from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import (
    load_scene_gt,
    load_scene_camera,
    HOPE_OBJECT_NAMES,
    YCBV_OBJECT_NAMES,
    OBJ_IDS,
    compute_t_id_log_err_pairs_for_object,
)
import json
import pinocchio as pin
import copy

def get_T_wo_drift(dataset, camera, obj_label):
    if obj_label == None:
        obj_label = list(dataset[0].keys())[0]
    T_cw_0:pin.SE3 = pin.SE3(camera[0]["T_cw"])
    T_co_0:pin.SE3 = pin.SE3(dataset[0][obj_label][0])
    T_wo_0:pin.SE3 = T_cw_0.inverse() * T_co_0
    ret = np.full((len(dataset)), None)
    for i in range(len(dataset)):
        T_cw:pin.SE3 = pin.SE3(camera[i]["T_cw"])
        T_co:pin.SE3 = pin.SE3(dataset[i][obj_label][0])
        T_wo:pin.SE3 = T_cw.inverse() * T_co
        T_oo:pin.SE3 = T_wo_0.inverse() * T_wo
        ret[i] = np.linalg.norm(T_oo.translation)
    return ret


def main():
    DATASET_ROOT = Path("/mnt/Data/HappyPose_Data/bop_datasets")
    max_frames = 330

    fig, ax = plt.subplots(2, figsize=(15,10))

    for i, scene in enumerate(["000048", "000049", "000050", "000051", "000052", "000053"]):
        ycbv_scene_path = DATASET_ROOT/"ycbv"/"test"/scene
        ycbv = load_scene_gt(ycbv_scene_path/"scene_gt.json", list(YCBV_OBJECT_NAMES.values()))
        ycbv_camera = load_scene_camera(ycbv_scene_path / "scene_camera.json")
        ycbv_drift = get_T_wo_drift(ycbv[:max_frames], ycbv_camera[:max_frames], None)
        ax[0].plot(np.arange(len(ycbv_drift)), ycbv_drift * 1000, "-", label=f"", linewidth=2)

    for i, scene in enumerate(["000001", "000002", "000003", "000004", "000005", "000006"]):
        hope_scene_path = DATASET_ROOT/"hopeVideo"/"test"/scene
        hope = load_scene_gt(hope_scene_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
        hope_camera = load_scene_camera(hope_scene_path / "scene_camera.json")
        hope_drift = get_T_wo_drift(hope[:max_frames], hope_camera[:max_frames], None)
        ax[1].plot(np.arange(len(hope_drift)), hope_drift * 1000, "-", label=f"", linewidth=7-i)

    fontsize = 17

    ax[0].set_ylim([-0.1, 10])
    ax[0].set_xlabel("frame", fontsize=fontsize)
    ax[0].set_ylabel(r"$|\bf t|$ [mm]", fontsize=fontsize)
    ax[0].set_title("YCB-V", fontsize=fontsize*1.2)
    ax[0].grid()

    ax[1].set_ylim([-0.1, 10])
    ax[1].set_xlabel("frame", fontsize=fontsize)
    ax[1].set_ylabel(r"$|\bf t|$ [mm]", fontsize=fontsize)
    ax[1].set_title("HOPE-Video", fontsize=fontsize*1.2)
    ax[1].grid()

    fig.subplots_adjust(hspace=0.35, top=0.90, bottom=0.1)
    fig.savefig(f"t_drift.png")
    fig.savefig(f"t_drift.svg")
    fig.savefig(f"t_drift.pdf")
    plt.show()

if __name__ == "__main__":
    main()