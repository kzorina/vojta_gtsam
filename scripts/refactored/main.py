import copy
from pathlib import Path
import json
import numpy as np
import pickle
import time
import utils
from utils import load_scene_camera, load_pickle, merge_T_cos_px_counts
from SamWrapper import SamWrapper
from GlobalParams import GlobalParams
from Vizualization_tools import display_factor_graph, animate_refinement, animate_state



def is_valid(Q, t_validity_treshold, R_validity_treshold):
    # R_det = np.linalg.det(Q[:3, :3]) ** 0.5
    # t_det = np.linalg.det(Q[3:6, 3:6]) ** 0.5
    R_det = np.linalg.det(Q[:3, :3]) ** (1/3)
    t_det = np.linalg.det(Q[3:6, 3:6]) ** (1/3)
    if t_det < t_validity_treshold and R_det < R_validity_treshold:
        return True
    return False

def recalculate_validity(refined_scene, t_validity_treshold, R_validity_treshold):
    recalculated_refined_scene = []
    for frame in range(len(refined_scene)):
        entry = {}
        for obj_label in refined_scene[frame]:
            entry[obj_label] = []
            for obj_idx in range(len(refined_scene[frame][obj_label])):
                track = copy.deepcopy(refined_scene[frame][obj_label][obj_idx])
                track["valid"] = is_valid(track["Q"], t_validity_treshold, R_validity_treshold)
                entry[obj_label].append(track)
        recalculated_refined_scene.append(entry)
    return recalculated_refined_scene


def refine_scene(scene_camera, frames_prediction, px_counts, params:GlobalParams):
    sam = SamWrapper(params)
    refined_scene = []
    for i in range(len(scene_camera)):
        time_stamp = i/30
        T_wc = np.linalg.inv(scene_camera[i]['T_cw'])
        Q_T_wc = np.eye(6)*10**(-6)
        detections = merge_T_cos_px_counts(frames_prediction[i], px_counts[i])  # T_co and Q for all detected object in a frame.
        sam.insert_detections({"T_wc":T_wc, "Q":Q_T_wc}, detections, time_stamp)
        current_state = sam.get_state()
        # animate_state(current_state, time_stamp)
        refined_scene.append(current_state.get_extrapolated_state(time_stamp, T_wc))
        # display_factor_graph(*utils.parse_variable_index(sam.tracks.factor_graph.isams[sam.tracks.factor_graph.active_chunk].getVariableIndex()))
        # time.sleep(1)
        print(f"\r({(i + 1)}/{len(scene_camera)})", end='')
    return refined_scene

def refine_dataset(DATASETS_PATH, DATASET_NAME, scenes, params):
    for scene_num in scenes:
        scene_path = DATASETS_PATH/DATASET_NAME/"test"/ f"{scene_num:06}"
        scene_camera = load_scene_camera(scene_path / "scene_camera.json")
        frames_prediction = load_pickle(scene_path / "frames_prediction.p")
        px_counts = load_pickle(scene_path / "frames_px_counts.p")
        refined_scene = refine_scene(scene_camera, frames_prediction, px_counts, params)
        # animate_refinement(refined_scene)
        recalculated_refined_scene = recalculate_validity(refined_scene, params.t_validity_treshold, params.R_validity_treshold)
        with open(scene_path / 'frames_refined_prediction.p', 'wb') as file:
            pickle.dump(recalculated_refined_scene, file)

def main():
    start_time = time.time()
    dataset_type = "hope"
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "SynthDynamicOcclusion"
    # scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scenes = [0]
    params = GlobalParams()

    refine_dataset(DATASETS_PATH, DATASET_NAME, scenes, params)

    refined_scene = load_pickle(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'frames_refined_prediction.p')
    scene_gt = utils.load_scene_gt(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'scene_gt.json', list(utils.HOPE_OBJECT_NAMES.values()))
    scene_camera = load_scene_camera(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}" / "scene_camera.json")
    animate_refinement(refined_scene, scene_gt, scene_camera)

    print(f"elapsed time: {time.time() - start_time:.2f}s")
    print(f"elapsed time: {utils.format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()
