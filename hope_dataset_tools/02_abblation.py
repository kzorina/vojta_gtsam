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
# method_names = {"ablation_synth_hope_dynamic_occlusion": "Ours (const. velocity)",
#                 "ablation_synth_hope_static": "Ours (const. pose)"}

method_names = {"ablation_synth_hope_dynamic_occlusion_5%": "Ours (const. velocity)",
                "ablation_synth_hope_static_5%": "Ours (const. pose)"}

for file_name in ["ablation_synth_hope_dynamic_occlusion_5%", "ablation_synth_hope_static_5%"]:
# for file_name in ["ablation_synth_hope_dynamic_occlusion", "ablation_synth_hope_static"]:
    # data = pd.read_csv(Path(__file__).parent / "abblation" / f"{file_name}.csv")
    # data = pd.read_csv("/home/vojta/PycharmProjects/gtsam_playground/hope_dataset_tools/ablation_synth_hope_dynamic_occlusion.csv", sep='\t')
    data = pd.read_csv(f"/home/vojta/PycharmProjects/gtsam_playground/hope_dataset_tools/{file_name}.csv", sep='\t')

    data = data.rename(
        columns={
            "mod": "mod",
            "max_derivative_order": "max_derivative_order",
            "max_track_age": "max_track_age",
            "cov_drift_lin_vel": "cov_drift_lin_vel",
            "cov_drift_ang_vel": "cov_drift_ang_vel",
            "outlier_rejection_treshold_trans": "outlier_trans",
            "outlier_rejection_treshold_rot": "outlier",
            "t_validity_treshold": "translation",
            "R_validity_treshold": "rotation",
            "bop19_average_recall": "recall",
            "bop19_average_precision": "precision",
        }
    )

    ind = data["mod"] == "cosypose"
    data_sam = data[~ind]
    data_cosypose = data[ind]
    data_sam["outlier"] = pd.to_numeric(data_sam["outlier"])
    data_sam = data_sam.sort_values(by=["outlier", "translation", "rotation"])

    d = data_sam
    outliers = np.unique(data_sam["outlier"])
    translations = np.unique(data_sam["translation"])
    rotations = np.unique(data_sam["rotation"])
    # mods = np.unique(data_sam["mod"])
    mods = np.unique(data_sam["mod"])

    # # plot outlier vs recal/precission for all thresholds
    # fig, ax = plt.subplots(
    #     1, 1, squeeze=True, figsize=(2 * 6.4, 4.8)
    # )  # type: plt.Figure, plt.Axes
    #
    # lw = 2
    # ci = 0
    # for i in range(len(translations)):
    #     for j in range(len(rotations)):
    #         sel = d[
    #             (d["translation"] == translations[i]) & (d["rotation"] == rotations[j])
    #         ]
    #         ax.plot(
    #             sel.outlier,
    #             sel.recall,
    #             "-o",
    #             label=f"Recall ({translations[i]:.1e},{rotations[j]})",
    #             # color=plt.colormaps["tab10"].colors[ci],
    #             lw=lw,
    #         )
    #         ax.plot(
    #             sel.outlier,
    #             sel.precision,
    #             "--x",
    #             label=f"Precision ({translations[i]:.1e},{rotations[j]})",
    #             # color=plt.colormaps["tab10"].colors[ci],
    #             lw=lw,
    #         )
    #         ci += 1
    #
    # # 0.7605166667	0.7720981387
    # ax.hlines(
    #     data_cosypose.recall,
    #     min(outliers),
    #     max(outliers),
    #     linestyles="-",
    #     label=f"Recall CosyPose [baseline]",
    #     # color=plt.colormaps["tab10"].colors[ci],
    #     lw=lw,
    # )
    # ax.hlines(
    #     data_cosypose.precision,
    #     min(outliers),
    #     max(outliers),
    #     linestyles="--",
    #     label=f"Precision CosyPose [baseline]",
    #     # color=plt.colormaps["tab10"].colors[ci],
    #     lw=lw,
    # )
    #
    # ax.set_xlabel("Outlier threshold")
    # ax.set_ylabel("Recall/Precision")
    # # ax.legend(
    # #     loc="upper center",
    # #     bbox_to_anchor=(0.5, 1.25),
    # #     ncol=4,
    # #     fancybox=True,
    # #     shadow=True,
    # # )
    # fig.subplots_adjust(top=0.8)
    # # fig.savefig("prec_recal_thresh.png")
    #
    # # Precision recall curve

    fig, ax = plt.subplots(
        1, 1, squeeze=True, figsize=(2 * 6.4, 4.8)
    )  # type: plt.Figure, plt.Axes

    # for outlier in outliers:
    ax.scatter(data_cosypose.recall, data_cosypose.precision, color="tab:orange", marker='x', s=80, linewidths=2,
               label="CosyPose", zorder=3)
    for ws in mods:
        # for t in translations:
        # d = data_sam[(data_sam["outlier"] == outlier) & (data_sam["translation"] == t)]
        d = data_sam[(data_sam["mod"] == ws)]
        # d = data_sam
        d = d.sort_values(by=["recall"])
        # d = data_sam[data_sam["outlier"] == outlier]
        if isinstance(ws, str):
            if ws.isnumeric():
                value = int(float(ws))
            else:
                value = ws
        else:
            value = ws
        if int(float(ws)) == 1:
            ax.plot(
                d.recall,
                d.precision,
                "-o",
                # label=rf"$\tau_\text{{outlier}}={outlier}, \tau_\text{{pred\_t}}={t}$",
                label=f"{method_names[file_name]}",
                zorder=2
            )
        else:
            # ax.plot(
            #     d.recall,
            #     d.precision,
            #     marker='o',
            #     markersize=10,
            #     linewidth=3,
            #     mfc='none',
            #     # label=rf"$\tau_\text{{outlier}}={outlier}, \tau_\text{{pred\_t}}={t}$",
            #     label=f"{method_names[file_name]}",
            # )
            colors = [None, None, "tab:green", "tab:purple"]
            names = [None, None, "recall oriented", "precision oriented"]
            ax.scatter(d.recall, d.precision, color=colors[int(float(ws))], marker='o', s=150, linewidths=2, facecolors='none', label=f"{names[int(float(ws))]}", zorder=3)

    # ax.plot(data_cosypose.precision, data_cosypose.recall, "x", label="CosyPose", ms=10)
    # ax.plot(data_cosypose.recall, data_cosypose.precision, "x", label="CosyPose", ms=10)


    fontsize = 13

    ax.set_xlabel("Recall", fontsize=fontsize)
    ax.set_ylabel("Precision", fontsize=fontsize)

    if "dynamic" in file_name:
        ax.set_title("Dynamic", fontsize=fontsize + 2)
    if "static" in file_name:
        ax.set_title("Static", fontsize=fontsize + 2)

    ax.set_xlim(0.0,1)
    ax.set_ylim(0.0,1)
    ax.axes.set_aspect("equal")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.grid()
    fig.subplots_adjust(top=0.8)
    fig.savefig(f"{file_name}.png")
    fig.savefig(f"{file_name}.svg")
    fig.savefig(f"{file_name}.pdf")

    plt.show()
