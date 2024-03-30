from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def main():
    data = pd.read_csv("/home/vojta/PycharmProjects/gtsam_playground/plots/mod_data2.csv", sep=',')
    data = data.sort_values(by=["modulo"])
    fig, ax = plt.subplots(2, 1)
    zero = data[data["method"] == "zero_velocity_cosypose"]
    const = data[data["method"] == "const_velocity_cosypose"]
    gtsam = data[data["method"] == "gtsam"]
    ax[0].plot(zero.modulo - 1, zero.recall,"-o",label=f"zero_velocity_cosypose")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)
    ax[0].plot(const.modulo - 1, const.recall,"-o",label=f"const_velocity_cosypose")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)
    ax[0].plot(gtsam.modulo - 1, gtsam.recall,"-o",label=f"gtsam")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)
    ax[1].plot(zero.modulo - 1, zero.precision,"-o",label=f"zero_velocity_cosypose")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)
    ax[1].plot(const.modulo - 1, const.precision,"-o",label=f"const_velocity_cosypose")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)
    ax[1].plot(gtsam.modulo - 1, gtsam.precision,"-o",label=f"gtsam")# color=plt.colormaps["tab10"].colors[ci],lw=lw,)

    ax[0].set_xlabel("frame gap size")
    ax[0].set_ylabel("Recall")

    ax[1].set_xlabel("frame gap size")
    ax[1].set_ylabel("Precision")

    ax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    plt.show()

if __name__ == "__main__":
    main()