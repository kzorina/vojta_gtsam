
from TracksWrapper import Tracks, Track
from GlobalParams import GlobalParams
import gtsam
from State import State, BareTrack

class SamWrapper:
    def __init__(self, params:GlobalParams):
        self.tracks:Tracks = Tracks(params)
        self.params:GlobalParams = params
        self.frame = 0
        pass

    def get_state(self):
        state = State(self.params)
        for obj_label in self.tracks.tracks:
            for track in self.tracks.tracks[obj_label]:
                bare_track = track.get_bare_track()
                state.add_bare_track(obj_label, bare_track)
        return state

    def insert_detections(self, T_wc_detection, T_co_detections, time_stamp):
        self.frame += 1
        self.tracks.camera.add_detection(T_wc_detection["T_wc"], T_wc_detection["Q"], time_stamp)
        T_wc = gtsam.Pose3(T_wc_detection["T_wc"])
        for obj_label in T_co_detections:
            matched_tracks = self.tracks.get_tracks_matches(obj_label, T_co_detections[obj_label], time_stamp)
            for i in range(len(T_co_detections[obj_label])):
                track:Track = matched_tracks[i]
                T_co = T_co_detections[obj_label][i]["T_co"]
                Q = T_co_detections[obj_label][i]["Q"]
                if track is None:
                    track:Track = self.tracks.create_track(obj_label)
                track.add_detection(T_wc, T_co, Q, time_stamp)
        self.tracks.remove_expired_tracks(time_stamp)
        self.tracks.factor_graph.update()
        if self.frame % self.params.chunk_size == 0:
            self.tracks.factor_graph.swap_chunks(self.tracks.tracks)
        self.tracks.update_stamp += 1
