from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

import gtsam
import scipy
import gtsam.utils.plot as gtsam_plot
from Vizualize import draw_3d_estimate

def main():
    my_cov = np.array([[0.1, 0, 0, 0, 0, 0],
                       [0, 0.1, 0, 0, 0, 0],
                       [0, 0, 0.1, 0, 0, 0],
                       [0, 0, 0, 0.5, 0, 0],
                       [0, 0, 0, 0, 0.5, 0],
                       [0, 0, 0, 0, 0, 0.1]])
    LANDMARK_NOISE:gtsam.noiseModel.Diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.1]))
    noise: gtsam.noiseModel.Gaussian = gtsam.noiseModel.Gaussian.Covariance(my_cov)
    c = noise.covariance()
    b = LANDMARK_NOISE.covariance()
    print(c)
    print(b)

if __name__ == "__main__":
    main()