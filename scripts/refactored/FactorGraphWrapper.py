import gtsam
from GlobalParams import GlobalParams
import numpy as np
from gtsam.symbol_shorthand import L, X, V

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
        a = np.array(factor.keys()) - L(0)
        if np.isin(2010001, a):
            print('')
        for i in range(self.chunk_count):
            self.active_graphs[i].add(factor)

    def insert_estimate(self, symbol, pose):
        if 2010001 == symbol - L(0):
            print('')
        self.initial_estimate.insert(symbol, pose)

    def swap_chunks(self, tracks):
        new_graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        for obj_label in tracks:
            for track in tracks[obj_label]:
                T_wo = track.get_T_wo_init()
                Q = track.get_Q_derivative(0)
                noise = gtsam.noiseModel.Gaussian.Covariance(Q)
                new_graph.add(gtsam.PriorFactorPose3(track.get_symbol(), T_wo, noise))
                initial_estimate.insert(track.get_symbol(), T_wo)
                for derivative_symbol in track.get_active_derivative_symbols():
                    derivative = self.current_estimate.atVector(derivative_symbol)
                    Q_derivative = self.isams[self.active_chunk].marginalCovariance(derivative_symbol)
                    derivative_noise = gtsam.noiseModel.Gaussian.Covariance(Q_derivative)
                    new_graph.add(gtsam.PriorFactorVector(derivative_symbol, derivative, derivative_noise))
                    initial_estimate.insert(derivative_symbol, derivative)
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
