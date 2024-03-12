#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-29
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# for file_name in ["abblation_synth_hope_static", "abblation_hope_video"]:
for file_name in ["ablation_synth_hope_dynamic_occlusion"]:
    # data = pd.read_csv(Path(__file__).parent / "abblation" / f"{file_name}.csv")
    data = pd.read_csv("/home/vojta/PycharmProjects/gtsam_playground/hope_dataset_tools/ablation_synth_hope_dynamic_occlusion.csv", sep='\t')
    data = data.rename(
        columns={
            "covariance1": "cov1",
            "covariance2": "cov2",
            "outlier_threshold": "outlier",
            "translation_validity_threshold": "translation",
            "Rotation_validity_threshold": "rotation",
            "bop19_average_recall": "recall",
            "bop19_average_precision": "precision",
        }
    )

    ind = data["cov1"] == "cosypose"
    data_sam = data[~ind]
    data_cosypose = data[ind]
    data_sam["outlier"] = pd.to_numeric(data_sam["outlier"])
    data_sam = data_sam.sort_values(by=["outlier", "translation", "rotation"])

    d = data_sam
    outliers = np.unique(data_sam["outlier"])
    translations = np.unique(data_sam["translation"])
    rotations = np.unique(data_sam["rotation"])

    # plot outlier vs recal/precission for all thresholds
    fig, ax = plt.subplots(
        1, 1, squeeze=True, figsize=(2 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes

    lw = 2
    ci = 0
    for i in range(len(translations)):
        for j in range(len(rotations)):
            sel = d[
                (d["translation"] == translations[i]) & (d["rotation"] == rotations[j])
            ]
            ax.plot(
                sel.outlier,
                sel.recall,
                "-o",
                label=f"Recall ({translations[i]:.1e},{rotations[j]:.0e})",
                # color=plt.colormaps["tab10"].colors[ci],
                lw=lw,
            )
            ax.plot(
                sel.outlier,
                sel.precision,
                "--x",
                label=f"Precision ({translations[i]:.1e},{rotations[j]:.0e})",
                # color=plt.colormaps["tab10"].colors[ci],
                lw=lw,
            )
            ci += 1

    # 0.7605166667	0.7720981387
    ax.hlines(
        data_cosypose.recall,
        min(outliers),
        max(outliers),
        linestyles="-",
        label=f"Recall CosyPose [baseline]",
        # color=plt.colormaps["tab10"].colors[ci],
        lw=lw,
    )
    ax.hlines(
        data_cosypose.precision,
        min(outliers),
        max(outliers),
        linestyles="--",
        label=f"Precision CosyPose [baseline]",
        # color=plt.colormaps["tab10"].colors[ci],
        lw=lw,
    )

    ax.set_xlabel("Outlier threshold")
    ax.set_ylabel("Recall/Precision")
    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.25),
    #     ncol=4,
    #     fancybox=True,
    #     shadow=True,
    # )
    fig.subplots_adjust(top=0.8)
    # fig.savefig("prec_recal_thresh.png")

    # Precision recall curve

    fig, ax = plt.subplots(
        1, 1, squeeze=True, figsize=(2 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes
    for outlier in outliers:
        for t in translations:
            d = data_sam[
                (data_sam["outlier"] == outlier) & (data_sam["translation"] == t)
            ]
            d = d.sort_values(by=["recall"])
            # d = data_sam[data_sam["outlier"] == outlier]
            ax.plot(
                d.precision,
                d.recall,
                "-o",
                label=rf"$\tau_\text{{outlier}}={outlier}, \tau_\text{{pred\_t}}={t}$",
            )
    ax.plot(data_cosypose.precision, data_cosypose.recall, "x", label="CosyPose", ms=10)

    ax.set_ylabel("Recall")
    ax.set_xlabel("Precision")
    ax.set_xlim(0.5,1)
    ax.axes.set_aspect("equal")

    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.25),
    #     ncol=4,
    #     fancybox=True,
    #     shadow=True,
    # )
    fig.subplots_adjust(top=0.8)
    fig.savefig(f"{file_name}.png")

    plt.show()

#
# # Plot recal precission for single outlier
# recall = np.zeros((len(translations), len(rotations)))
# precision = np.zeros_like(recall)
# xx, yy = np.zeros_like(recall), np.zeros_like(recall)
# d = data_sam[data_sam["outlier"] == 5]
# for i in range(len(translations)):
#     for j in range(len(rotations)):
#         sel = d[(d["translation"] == translations[i]) & (d["rotation"] == rotations[j])]
#         xx[i, j] = sel.translation
#         yy[i, j] = sel.rotation
#         recall[i, j] = sel.recall
#         precision[i, j] = sel.precision
#
# fig: plt.Figure
# fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(6.4 * 2, 4.8))
# ax = axes[0]
# pc = ax.pcolormesh(xx, yy, recall, cmap="winter")
# fig.colorbar(pc)
# ax.set_xticks(translations)
# ax.set_xlabel("Translation threshold")
# ax.set_yticks(rotations)
# ax.set_ylabel("Rotation threshold")
# ax.set_title("Avg. recall")
# ax.grid(False)
# ax = axes[1]
#
# pc = ax.pcolormesh(xx, yy, precision, cmap="winter")
# fig.colorbar(pc)
# ax.set_xticks(translations)
# ax.set_xlabel("Translation threshold")
# ax.set_yticks(rotations)
# ax.set_ylabel("Rotation threshold")
# ax.set_title("Avg. precision")
# ax.grid(False)
# fig.suptitle("SynthHopeStatic; outlier threshold = 5")
# fig.savefig("outlier_5.png")

# plt.show()
