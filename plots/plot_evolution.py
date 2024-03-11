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

frames_refined_prediction = pickle.load(open(ds_path / "frames_refined_prediction.p", "rb"))
frames_prediction = pickle.load(open(ds_path / "frames_prediction.p", "rb"))
scene_gt = load_scene_gt(ds_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
camera_poses = json.load(open(ds_path / "scene_camera.json"))
#
# frames_prediction = pickle.load(open("frames_prediction.p", "rb"))
# frames_refined_prediction = pickle.load(open("frames_refined_prediction3.p", "rb"))

all_object_labels = set()
for frame in frames_prediction:
    all_object_labels.update(frame.keys())

all_object_labels = sorted(all_object_labels)
# for obj_label in all_object_labels:
for obj_label in ['BBQSauce']:
    fig, ax = plt.subplots(
        1, 1, squeeze=True, figsize=(3 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes
    ax.set_title(f"Object label: {obj_label}")
    v = compute_t_id_log_err_pairs_for_object(
        frames_prediction, obj_label, camera_poses
    )

    ax.plot(
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
        conf = np.bitwise_and(v[:, 5] < 0.000005,  v[:, 6] < 0.001)
        # conf = np.bitwise_and(v[:, 5] < 1.00000,  v[:, 6] < 1.000)
        # conf = np.bitwise_and(conf, v[:, 3] > .5)
        mask_predicted = np.bitwise_and(v[:, 1] == track_id, conf)
        mask = mask_predicted
        # err = np.clip(1e15*v[mask, 3], 0, 1)
        # ax.fill_between(
        #     v[mask, 0],
        #     v[mask, 2] + err,
        #     v[mask, 2] - err,
        #     alpha=0.5,
        #     color=plt.colormaps["tab10"].colors[i + 1],
        # )
        print(int(track_id))
        ax.plot(
            v[mask, 0],
            v[mask, 2],
            label=f"SAMPose track {int(track_id)}",
            color=plt.colormaps["tab10"].colors[(i + 1) % 10],
            linestyle="-",
            marker="o",
        )

    # plot uncertainties
    # fig1, axes = plt.subplots(
    #     1, 2, squeeze=False, sharex=True, sharey=False
    # )
    # axes[0, 0].set_title(f"det(Qt)^2")
    # axes[0, 1].set_title(f"det(Qr)^2")
    # for i, track_id in enumerate(np.unique(v[:, 1])):
    #     mask = v[:, 1] == track_id
    #     axes[0, 0].plot(
    #         v[mask, 0],
    #         v[mask, 5],
    #         color=plt.colormaps["tab10"].colors[i + 1],
    #     )
    #     axes[0, 1].plot(
    #         v[mask, 0],
    #         v[mask, 6],
    #         color=plt.colormaps["tab10"].colors[i + 1],
    #     )

    v = compute_t_id_log_err_pairs_for_object(scene_gt, obj_label, camera_poses)
    if len(v.shape) == 1:
        plt.close(fig)
        continue
    ax.plot(
        v[:, 0],
        v[:, 2],
        label="GT",
        color="k",
        linestyle="",
        marker="x",
        ms=10,
        alpha=0.5,
    )

    ax.set_xlabel("Frame id")
    ax.set_ylabel("||log(T)||")
    ax.legend()
    fig.savefig(f"evolution_{obj_label}.png")

    plt.show()
