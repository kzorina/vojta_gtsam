import gtsam
import numpy as np
from scipy.special import factorial
from ScenePlotter import Plotter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from collections import defaultdict

class BareTrack:
    def __init__(self, idx, T_wo:gtsam.Pose3, Q:np.ndarray, derivatives:np.ndarray, Q_derivatives:np.ndarray, time_stamp:float, extrapolation_function=None):
        assert derivatives.shape[1] == 6
        assert Q_derivatives.shape[1:] == (6, 6)
        self.idx = idx
        self.T_wo = T_wo
        self.derivatives = derivatives
        self.Q = Q
        self.Q_derivatives = Q_derivatives
        self.time_stamp = time_stamp
        self.order = self.derivatives.shape[0]
        self.extrapolation_function = BareTrack.extrapolate_func

    @staticmethod
    def extrapolate_func(dt:float, T_wo:gtsam.Pose3, derivatives:np.ndarray, Q:np.ndarray, Q_derivatives:np.ndarray):
        order = derivatives.shape[0]

        dts = np.full((order), dt)
        i = np.arange((order)) + 1
        ret = (dts**i)/factorial(i)
        x = ret[np.newaxis].T

        tau = np.sum(derivatives * x, axis=0)
        new_Q = Q + np.sum(Q_derivatives * x[:, np.newaxis], axis=0)
        R_new = gtsam.Rot3.Expmap(tau[:3]) * T_wo.rotation()
        t_new = T_wo.translation() + tau[3:6]
        new_T_wo = gtsam.Pose3(R_new, t_new)
        return new_T_wo, new_Q

    def extrapolate(self, time_stamp):
        dt = (time_stamp - self.time_stamp)
        new_T_wo, new_Q = self.extrapolation_function(dt, self.T_wo, self.derivatives, self.Q, self.Q_derivatives)
        # if dt > 0.001:
        #     print('')
        return new_T_wo, new_Q

class State:
    def __init__(self):
        self.bare_tracks = defaultdict(list)
        pass

    def add_bare_track(self, obj_label, bare_track):
        self.bare_tracks[obj_label].append(bare_track)

    def get_extrapolated_state(self, time_stamp, T_wc:gtsam.Pose3):
        ret = {}
        T_wc = gtsam.Pose3(T_wc)
        for obj_label in self.bare_tracks:
            ret[obj_label] = []
            for bare_track in self.bare_tracks[obj_label]:
                T_wo, Q = bare_track.extrapolate(time_stamp)
                T_co: gtsam.Pose3 = gtsam.Pose3(T_wc).inverse() * T_wo
                idx = bare_track.idx
                ret[obj_label].append({"T_wo": T_wo.matrix(),
                                       "T_wc": T_wc.matrix(),
                                       "T_co": T_co.matrix(),
                                       "id": idx,
                                       "Q": Q,
                                       "valid": None})
        return ret


if __name__ == "__main__":
    b = BareTrack("bagr", gtsam.Pose3(np.eye(4)), np.eye(6)*0.01, np.vstack([np.ones(6)*0.1, np.ones(6), np.ones(6)]), np.stack([np.eye(6)*0.1, np.eye(6), np.eye(6)]), 0.0)
    T_wo, Q = b.extrapolate(0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotter = Plotter(ax)

    def update_view(val):
        plotter.clear()
        plotter.reset_default_lim()
        dt = slider.val
        T_wo, Q = b.extrapolate(dt)
        plotter.plot_Q(Q[3:6, 3:6], T_wo)
        plotter.plot_T(T_wo)

    axhauteur = plt.axes([0.2, 0.1, 0.65, 0.03])



    slider = Slider(axhauteur, 'frame', 0, 1, valinit=0)
    slider.on_changed(update_view)
    plt.show()
