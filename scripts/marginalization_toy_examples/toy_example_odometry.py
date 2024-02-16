import gtsam
import numpy as np
from gtsam import Symbol
from gtsam.symbol_shorthand import B, V, X, L
from typing import List, Dict, Set
import pinocchio as pin
import graphviz

# import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
def rnd_T():
    T_wc = pin.SE3.Random()
    T_wc_gtsam = gtsam.Pose3(T_wc.homogeneous)
    return T_wc_gtsam

def marginalize_key(isam, keys):
    leafKeys = gtsam.KeyList()
    for key in keys:
        leafKeys.push_front(key)
    isam.marginalizeLeaves(leafKeys)

def main():
    new_graph = gtsam.NonlinearFactorGraph()

    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.1)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)
    noise = gtsam.noiseModel.Gaussian.Covariance(np.eye(6))
    current_frame = 0
    initial_estimate = gtsam.Values()

    #######

    T_bc = rnd_T()
    new_graph.add(gtsam.PriorFactorPose3(X(0), T_bc, noise))
    initial_estimate.insert(X(0), T_bc)

    for i in range(1, 10):  # odometry
        current_frame += 1
        new_graph.add(gtsam.BetweenFactorPose3(X(i-1), X(i), gtsam.Pose3(np.eye(4)), noise))
        initial_estimate.insert(X(i), T_bc)

        isam.update(new_graph, initial_estimate)
        current_estimate = isam.calculateEstimate()
        initial_estimate.clear()
        new_graph = gtsam.NonlinearFactorGraph()

    print(f"{isam.getVariableIndex()}")
    graphviz.Source(isam.dot(), filename=f'bayesTree{current_frame}_{0}', format='svg').view()
    marginalize_key(isam, [X(0), X(1), X(2), X(3), X(4), X(5)])
    # marginalize_key(isam, [X(5)])
    isam.update(new_graph, initial_estimate)
    print(f"{isam.getVariableIndex()}")
    graphviz.Source(isam.dot(), filename=f'bayesTree{current_frame}_{1}', format='svg').view()






if __name__ == "__main__":
    main()

