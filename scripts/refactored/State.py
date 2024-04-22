import gtsam
import numpy as np
from scipy.special import factorial
from ScenePlotter import Plotter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from collections import defaultdict
from GlobalParams import GlobalParams

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
        ret = ((dts**i)/factorial(i))
        x = ret[np.newaxis].T

        tau = np.sum(derivatives * x, axis=0)
        n = order - 1
        new_Q = Q + np.sum(Q_derivatives[:-1] * x[:-1][:, np.newaxis]**2, axis=0) + Q_derivatives[-1] * dt ** (2*(n) + 1)/((factorial(n)**2) * (2*(n) + 1))
        R_new = gtsam.Rot3.Expmap(tau[:3]) * T_wo.rotation()
        t_new = T_wo.translation() + tau[3:6]
        new_T_wo = gtsam.Pose3(R_new, t_new)

        return new_T_wo, new_Q
        vel = derivatives[0, :]
        acc = derivatives[1, :]
        Q_vel = Q_derivatives[0, :, :]
        Q_acc = Q_derivatives[1, :, :]
        test_R_new = gtsam.Rot3.Expmap(vel[:3] * dt + (acc[:3] * dt ** 2) / 2) * T_wo.rotation()
        test_t_new = T_wo.translation() + vel[3:6] * dt + (acc[3:6] * dt ** 2) / 2
        test_T_wo_new = gtsam.Pose3(test_R_new, test_t_new)
        Q_vel_new = Q_vel + Q_acc*dt

        test_Q_new = Q + (Q_vel * dt ** 2) + (Q_acc * dt ** 3) / 3
        # test_Q_new = Q + Q_vel_new * dt ** 2
        return test_T_wo_new, test_Q_new

    def extrapolate(self, time_stamp):
        # dt = (time_stamp - self.time_stamp)
        dt = (time_stamp - self.time_stamp) * 1  # TODO: CHANGE THIS BAcK
        new_T_wo, new_Q = self.extrapolation_function(dt, self.T_wo, self.derivatives, self.Q, self.Q_derivatives)
        # if dt > 0.001:
        #     print('')
        return new_T_wo, new_Q

class State:
    def __init__(self, params:GlobalParams):
        self.bare_tracks = defaultdict(list)
        self.params:GlobalParams = params
        pass

    def add_bare_track(self, obj_label, bare_track):
        self.bare_tracks[obj_label].append(bare_track)

    @staticmethod
    def is_valid(Q, t_validity_treshold, R_validity_treshold):
        R_det = np.linalg.det(Q[:3, :3]) ** 0.5
        t_det = np.linalg.det(Q[3:6, 3:6]) ** 0.5
        # R_det = np.linalg.det(Q[:3, :3]) ** (1 / 3)
        # t_det = np.linalg.det(Q[3:6, 3:6]) ** (1 / 3)
        if t_det < t_validity_treshold and R_det < R_validity_treshold:
            return True
        return False

    def get_extrapolated_state(self, time_stamp, T_wc:gtsam.Pose3):
        ret = {}
        T_wc = gtsam.Pose3(T_wc)
        for obj_label in self.bare_tracks:
            ret[obj_label] = []
            for bare_track in self.bare_tracks[obj_label]:
                T_wo, Q = bare_track.extrapolate(time_stamp)
                T_co: gtsam.Pose3 = gtsam.Pose3(T_wc).inverse() * T_wo
                idx = bare_track.idx

                validity = State.is_valid(Q, self.params.t_validity_treshold, self.params.R_validity_treshold)

                #  remove overlapping discrete symmetries
                if validity == True and self.params.reject_overlaps > 0:
                    for obj_inst in range(len(ret[obj_label])):
                        if ret[obj_label][obj_inst]["valid"]:
                            other_T_wo = ret[obj_label][obj_inst]["T_wo"]
                            other_Q = ret[obj_label][obj_inst]["Q"]
                            dist = np.linalg.norm(T_wo.matrix()[:3, 3] - other_T_wo[:3, 3])
                            if dist < self.params.reject_overlaps:
                                if np.linalg.det(Q) > np.linalg.det(other_Q):
                                    validity = False
                                else:
                                    ret[obj_label][obj_inst]["valid"] = False

                ret[obj_label].append({"T_wo": T_wo.matrix(),
                                       "T_wc": T_wc.matrix(),
                                       "T_co": T_co.matrix(),
                                       "id": idx,
                                       "Q": Q,
                                       "valid": validity})
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
