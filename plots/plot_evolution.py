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
import pinocchio as pin

from utils import (
    load_scene_gt,
    HOPE_OBJECT_NAMES,
    OBJ_IDS,
    compute_t_id_log_err_pairs_for_object,
)

DATASET_ROOT = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")

ds_path = DATASET_ROOT / "SynthDynamicOcclusion" / "test" / "000000"
# ds_path = DATASET_ROOT / "SynthStatic" / "test" / "000000"

frames_refined_prediction = pickle.load(open(ds_path / "frames_refined_prediction.p", "rb"))
frames_prediction = pickle.load(open(ds_path / "frames_prediction.p", "rb"))
scene_gt = load_scene_gt(ds_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
camera_poses = json.load(open(ds_path / "scene_camera.json"))

# frames_prediction = pickle.load(open("frames_prediction.p", "rb"))
# frames_refined_prediction = pickle.load(open("frames_refined_prediction3.p", "rb"))

all_object_labels = set()
for frame in frames_prediction:
    all_object_labels.update(frame.keys())

all_object_labels = sorted(all_object_labels)
# for obj_label in all_object_labels:
# for obj_label in ['BBQSauce', 'Cookies']:
for obj_label in all_object_labels:
    fig, ax = plt.subplots(
        2, 1, squeeze=True, figsize=(3 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes
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
        continue
    for i, track_id in enumerate(np.unique(v[:, 1])):
        # conf = np.bitwise_and(v[:, 5] < 0.0000025,  v[:, 6] < 0.0005)
        conf = v[:, 4] > .5

        mask_predicted = np.bitwise_and(v[:, 1] == track_id, conf)

        mask = mask_predicted

        print(int(track_id))
        ax[0].plot(
            v[mask, 0],
            v[mask, 2],
            label=f"SAMPose track {int(track_id)}",
            color=plt.colormaps["tab10"].colors[(i + 1) % 10],
            linestyle="-",
            marker="o",
        )

    v = compute_t_id_log_err_pairs_for_object(scene_gt, obj_label, camera_poses)
    if len(v.shape) == 1:
        plt.close(fig)
        continue
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
    v = compute_t_id_log_err_pairs_for_object(
        frames_refined_prediction, obj_label, camera_poses
    )

    if len(v.shape) == 1:
        plt.close(fig)
        continue
    for i, track_id in enumerate(np.unique(v[:, 1])):
        # conf = np.bitwise_and(v[:, 5] < 0.0000025,  v[:, 6] < 0.0005)
        conf = v[:, 4] > .5

        mask_predicted = np.bitwise_and(v[:, 1] == track_id, conf)

        mask = mask_predicted

        print(int(track_id))
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
        continue
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
    ax[1].set_xlabel("Frame id")
    ax[1].set_ylabel("||T.translation||")

    # ax.legend()
    fig.savefig(f"evolution_{obj_label}.png")

    plt.show()
