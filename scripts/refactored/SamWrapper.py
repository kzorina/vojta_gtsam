
from TracksWrapper import Tracks, Track
from GlobalParams import GlobalParams
import gtsam

class SamWrapper:
    def __init__(self, params:GlobalParams):
        self.tracks:Tracks = Tracks(params)
        self.params:GlobalParams = params
        self.frame = 0
        pass

    def get_state(self):
        ret = {}
        for obj_label in self.tracks.tracks:
            ret[obj_label] = []
            for track in self.tracks.tracks[obj_label]:
                T_wo = track.get_T_wo_init()
                Q = track.get_Q_init()
                idx = track.idx
                ret[obj_label].append({"T_wo":T_wo, "id":idx, "Q":Q})
        return ret

    def get_state_extrapolated(self, T_wc, time_stamp):
        ret = {}
        for obj_label in self.tracks.tracks:
            ret[obj_label] = []
            for track in self.tracks.tracks[obj_label]:
                T_wo = track.get_T_wo_extrapolated(time_stamp)
                T_co:gtsam.Pose3 = gtsam.Pose3(T_wc).inverse() * T_wo
                Q = track.get_Q_extrapolated(time_stamp)
                idx = track.idx
                ret[obj_label].append({"T_co":T_co.matrix(), "id":idx, "Q":Q, "valid": track.is_valid(time_stamp)})
        return ret

    def insert_detections(self, T_wc_detection, T_co_detections, time_stamp):
        self.frame += 1
        self.tracks.camera.add_detection(T_wc_detection["T_wc"], T_wc_detection["Q"], time_stamp)
        for obj_label in T_co_detections:
            matched_tracks = self.tracks.get_tracks_matches(obj_label, T_co_detections[obj_label])
            for i in range(len(T_co_detections[obj_label])):
                track:Track = matched_tracks[i]
                T_co = T_co_detections[obj_label][i]["T_co"]
                Q = T_co_detections[obj_label][i]["Q"]
                if track is None:
                    track:Track = self.tracks.create_track(obj_label)
                track.add_detection(T_co, Q, time_stamp)
        self.tracks.remove_expired_tracks()
        self.tracks.factor_graph.update()
        if self.frame % self.params.chunk_size == 0:
            self.tracks.factor_graph.swap_chunks(self.tracks.tracks)
        self.tracks.update_stamp += 1
