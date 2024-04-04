import gtsam
from GlobalParams import GlobalParams
import numpy as np

class FactorGraphWrapper:
    def __init__(self, params:GlobalParams):
        self.params = params
        self.marginals = None
        self.chunk_count = 2
        self.active_chunk = 0
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        # self.parameters.relinearizeSkip = 1
        self.isams = []
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.new_graph = gtsam.NonlinearFactorGraph()
        self.active_graphs = []
        for i in range(self.chunk_count):
            self.isams.append(gtsam.ISAM2(self.parameters))
            self.active_graphs.append(gtsam.NonlinearFactorGraph())


    def add_factor(self, factor):
        self.new_graph.add(factor)
        for i in range(self.chunk_count):
            self.active_graphs[i].add(factor)

    def inser_estimate(self, symbol, pose):
        self.initial_estimate.insert(symbol, pose)

    def swap_chunks(self, tracks):
        new_graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        for obj_label in tracks:
            for track in tracks[obj_label]:
                T_wo = track.get_T_wo_init()
                Q = track.get_Q_init()
                noise = gtsam.noiseModel.Gaussian.Covariance(Q)
                new_graph.add(gtsam.PriorFactorPose3(track.get_symbol(), T_wo, noise))
                initial_estimate.insert(track.get_symbol(), T_wo)
                velocity_symbol = track.get_nu_symbol()
                if track.num_detected >= 2:
                    nu = self.current_estimate.atVector(velocity_symbol)
                    Q_velocity = self.isams[self.active_chunk].marginalCovariance(velocity_symbol)
                else:
                    nu = np.zeros(6)
                    Q_velocity = np.eye(6)*10
                velocity_noise = gtsam.noiseModel.Gaussian.Covariance(Q_velocity)
                new_graph.add(gtsam.PriorFactorVector(velocity_symbol, nu, velocity_noise))
                initial_estimate.insert(velocity_symbol, nu)
        self.active_graphs[self.active_chunk] = new_graph.clone()
        self.isams[self.active_chunk] = gtsam.ISAM2(self.parameters)
        self.isams[self.active_chunk].update(new_graph, initial_estimate)
        self.active_chunk = (self.active_chunk + 1) % self.chunk_count

    def update(self):
        for isam in self.isams:
            isam.update(self.new_graph, self.initial_estimate)
        self.current_estimate = self.isams[self.active_chunk].calculateEstimate()
        self.initial_estimate.clear()
        self.new_graph = gtsam.NonlinearFactorGraph()
        # print(self.active_chunk)
        # self.marginals = gtsam.Marginals(self.active_graphs[self.active_chunk], self.current_estimate)

    def marginalCovariance(self, symbol):
        return self.isams[self.active_chunk].marginalCovariance(symbol)
        # return self.marginals.marginalCovariance(symbol)
