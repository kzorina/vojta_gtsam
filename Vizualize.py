import matplotlib.pyplot as plt

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Symbol
import numpy as np

np.set_printoptions(precision=3)
count = 0
def draw_3d_estimate(graph: gtsam.NonlinearFactorGraph, current_estimate: gtsam.Values, lines=True):
    """Display the current estimate of a factor graph"""
    global count
    # Compute the marginals for all states in the graph.
    marginals = gtsam.Marginals(graph, current_estimate)
    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    if not fig.axes:
        axes = fig.add_subplot(projection='3d')
    else:
        axes = fig.axes[0]
    plt.cla()

    for i in graph.keyVector():
        current_pose = current_estimate.atPose3(i)
        # cg = graph.at(i)
        name = str(Symbol(i).string())
        c = marginals.marginalCovariance(i)
        gtsam_plot.plot_pose3(0, current_pose, 0.2, marginals.marginalCovariance(i))
        # print(str(Symbol(i).string()), marginals.marginalCovariance(i))
        axes.text(current_pose.x(), current_pose.y(), current_pose.z(), str(Symbol(i).string()), fontsize=15)

    if lines:
        for i in range(graph.nrFactors()):

            factor: gtsam.gtsam.BetweenFactorPose3 = graph.at(i)
            keys = factor.keys()
            if len(keys) == 2:
                s_pose = current_estimate.atPose3(keys[0])
                e_pose = current_estimate.atPose3(keys[1])
                axes.plot(xs=[s_pose.x(), e_pose.x()], ys=[s_pose.y(), e_pose.y()], zs=[s_pose.z(), e_pose.z()], color="purple")

    ranges = (-0.8, 0.8)
    axes.set_xlim3d(ranges[0], ranges[1])
    axes.set_ylim3d(ranges[0], ranges[1])
    axes.set_zlim3d(ranges[0], ranges[1])
    fig.savefig(f'images/{count}.png')
    count += 1
    plt.pause(0.05)