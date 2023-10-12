
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
from Vizualize import draw_3d_estimate


if __name__ == '__main__':
    noise1:gtsam.gtsam.noiseModel.Diagonal  = gtsam.noiseModel.Diagonal.Sigmas(np.array([1 * np.pi / 180,
                                                             1 * np.pi / 180,
                                                             1 * np.pi / 180,
                                                             1,
                                                             3,
                                                             2]))
    noise2:gtsam.gtsam.noiseModel.Isotropic = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    noise2: gtsam.gtsam.noiseModel.Gaussian = gtsam.noiseModel.Gaussian.(3, 0.1)

    print(noise1.covariance())
    # print(type(noise2.covariance()))
    # print(noise2.covariance())