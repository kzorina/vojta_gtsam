#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-27
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.widgets import Slider, Button, RadioButtons
import pinocchio as pin

from utils import (
    load_scene_gt,
    HOPE_OBJECT_NAMES,
    OBJ_IDS,
    compute_t_id_log_err_pairs_for_object,
)

DATASET_ROOT = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")

ds_path = DATASET_ROOT / "SynthDynamicOcclusion" / "test" / "000002"
# ds_path = DATASET_ROOT / "SynthStatic" / "test" / "000000"

frames_refined_prediction = pickle.load(open(ds_path / "frames_refined_prediction.p", "rb"))
frames_prediction = pickle.load(open(ds_path / "frames_prediction.p", "rb"))
scene_gt = load_scene_gt(ds_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
camera_poses = json.load(open(ds_path / "scene_camera.json"))

# frames_prediction = pickle.load(open("frames_prediction.p", "rb"))
# frames_refined_prediction = pickle.load(open("frames_refined_prediction3.p", "rb"))

all_object_labels = set()
for frame in scene_gt:
    all_object_labels.update(frame.keys())

all_object_labels = sorted(all_object_labels)
# for obj_label in all_object_labels:
# for obj_label in ['BBQSauce', 'Cookies']:

def refresh_plot(num):
    ax[0].clear()

    ax[1].clear()
    trt = slider1.val
    rrt = slider2.val
    ax[0].set_title(f"Object label: {obj_label}")
    v = compute_t_id_log_err_pairs_for_object(
        frames_prediction, obj_label, camera_poses
    )

    ax[0].plot(
        v[:, 0],
        v[:, 2],
        label="CosyPose",
        color="tab:blue",
        linestyle="",
        marker="o",
        ms=10,
    )
    v = compute_t_id_log_err_pairs_for_object(
        frames_refined_prediction, obj_label, camera_poses
    )

    if len(v.shape) == 1:
        plt.close(fig)
    for i, track_id in enumerate(np.unique(v[:, 1])):
        conf = np.bitwise_and(v[:, 6] < trt,  v[:, 7] < rrt)
        # conf = v[:, 4] > .5

        mask_predicted = np.bitwise_and(v[:, 1] == track_id, conf)

        mask = mask_predicted

        ax[0].plot(
            v[mask, 0],
            v[mask, 2],
            label=f"SAMPose track {int(track_id)}",
            color=plt.colormaps["tab10"].colors[(i + 1) % 10],
            linestyle="-",
            marker="o",
        )

    ax[0].set_xlabel("Frame id")
    ax[0].set_ylabel("||log3(T.rotation)||")

    ax[1].set_title(f"Object label: {obj_label}")
    v = compute_t_id_log_err_pairs_for_object(
        frames_prediction, obj_label, camera_poses
    )

    ax[1].plot(
        v[:, 0],
        v[:, 3],
        label="CosyPose",
        color="tab:blue",
        linestyle="",
        marker="o",
        ms=10,
    )

    v = compute_t_id_log_err_pairs_for_object(scene_gt, obj_label, camera_poses)
    if len(v.shape) == 1:
        plt.close(fig)

    ax[0].plot(
        v[:, 0],
        v[:, 2],
        label="GT",
        color="k",
        linestyle="",
        marker="x",
        ms=10,
        alpha=0.5,
    )

    ax[1].plot(
        v[:, 0],
        v[:, 3],
        label="GT",
        color="k",
        linestyle="",
        marker="x",
        ms=10,
        alpha=0.5,
    )



    v = compute_t_id_log_err_pairs_for_object(
        frames_refined_prediction, obj_label, camera_poses
    )

    if len(v.shape) == 1:
        plt.close(fig)



    for i, track_id in enumerate(np.unique(v[:, 1])):
        conf = np.bitwise_and(v[:, 6] < trt,  v[:, 7] < rrt)
        # conf = v[:, 4] > .5

        mask_predicted = np.bitwise_and(v[:, 1] == track_id, conf)

        mask = mask_predicted

        # print(int(track_id))
        ax[1].plot(
            v[mask, 0],
            v[mask, 3],
            label=f"SAMPose track {int(track_id)}",
            color=plt.colormaps["tab10"].colors[(i + 1) % 10],
            linestyle="-",
            marker="o",
        )

    v = compute_t_id_log_err_pairs_for_object(scene_gt, obj_label, camera_poses)
    if len(v.shape) == 1:
        plt.close(fig)

    ax[1].set_xlabel("Frame id")
    ax[1].set_ylabel("||T.translation||")

for obj_label in all_object_labels:

    fig, ax = plt.subplots(
        3, 1, squeeze=True, figsize=(3 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes
    axhauteur1 = plt.axes([0.2, 0.2, 0.65, 0.03])
    axhauteur2 = plt.axes([0.2, 0.15, 0.65, 0.03])
    tvt = 0.000005
    rvt = 0.00075
    slider1 = Slider(axhauteur1, 'tvt', tvt*0.1, tvt*10, valinit=tvt)
    slider2 = Slider(axhauteur2, 'rvt', rvt*0.1, rvt*10, valinit=rvt)
    slider1.on_changed(refresh_plot)
    slider2.on_changed(refresh_plot)
    refresh_plot(None)

    # ax.legend()
    # fig.savefig(f"evolution_{obj_label}.png")

    plt.show()
