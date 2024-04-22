import copy
from pathlib import Path
import json
import numpy as np
import pickle
import time
import utils
from utils import load_scene_camera, load_pickle, merge_T_cos_px_counts
from SamWrapper import SamWrapper
from State import *
from GlobalParams import GlobalParams
from Vizualization_tools import display_factor_graph, animate_refinement, animate_state
from bop_tools import convert_frames_to_bop, export_bop
import copy
import os
import shutil
import multiprocessing


def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

def recalculate_validity(results, t_validity_treshold, R_validity_treshold, reject_overlaps):
    recalculated_results = {}
    for result_key in results:
        refined_scene = results[result_key]
        recalculated_refined_scene = []
        for frame in range(len(refined_scene)):
            entry = {}
            for obj_label in refined_scene[frame]:
                entry[obj_label] = []
                for obj_idx in range(len(refined_scene[frame][obj_label])):
                    track = copy.deepcopy(refined_scene[frame][obj_label][obj_idx])

                    validity = State.is_valid(track["Q"], t_validity_treshold, R_validity_treshold)
                    T_wo = track["T_wo"]
                    Q = track["Q"]
                    #  remove overlapping discrete symmetries
                    if validity and reject_overlaps > 0:
                        for obj_inst in range(len(entry[obj_label])):
                            if entry[obj_label][obj_inst]["valid"]:
                                other_T_wo = entry[obj_label][obj_inst]["T_wo"]
                                other_Q = entry[obj_label][obj_inst]["Q"]
                                dist = np.linalg.norm(T_wo[:3, 3] - other_T_wo[:3, 3])
                                if dist < reject_overlaps:
                                    if np.linalg.det(Q) > np.linalg.det(other_Q):
                                        validity = False
                                    else:
                                        entry[obj_label][obj_inst]["valid"] = False

                    track["valid"] = validity
                    entry[obj_label].append(track)
            recalculated_refined_scene.append(entry)
        recalculated_results[result_key] = recalculated_refined_scene
    return recalculated_results


def refine_data(scene_camera, frames_prediction, px_counts, params:GlobalParams):
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

def refine_scene(scene_path, params):
    scene_camera = load_scene_camera(scene_path / "scene_camera.json")
    frames_prediction = load_pickle(scene_path / "frames_prediction.p")
    px_counts = load_pickle(scene_path / "frames_px_counts.p")
    refined_scene = refine_data(scene_camera, frames_prediction, px_counts, params)
    return refined_scene

def anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, params, dataset_type='hope'):
    results = {}
    print(f"scenes: {scenes}")
    for scene_num in scenes:
        scene_path = DATASETS_PATH/DATASET_NAME/"test"/ f"{scene_num:06}"
        refined_scene = refine_scene(scene_path, params)
        results[scene_num] = refined_scene
        with open(scene_path / 'frames_refined_prediction.p', 'wb') as file:
            pickle.dump(refined_scene, file)
    for tvt in [1]:
        for rvt in [0.1, 2.2, 2.5, 3, 3.5]:
            forked_params = copy.deepcopy(params)
            forked_params.R_validity_treshold = params.R_validity_treshold * rvt
            forked_params.t_validity_treshold = params.t_validity_treshold * tvt
            # forked_params.R_validity_treshold = rvt

            recalculated_results = recalculate_validity(results, forked_params.t_validity_treshold, forked_params.R_validity_treshold, forked_params.reject_overlaps)
            output_name = f'gtsam_{DATASET_NAME}-test_{str(forked_params)}_.csv'
            export_bop(convert_frames_to_bop(recalculated_results, dataset_type), DATASETS_PATH / DATASET_NAME / "ablation" / output_name)

def main():
    start_time = time.time()
    dataset_type = "hope"
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "SynthDynamicOcclusion"
    # DATASET_NAME = "SynthStatic"
    # DATASET_NAME = "hopeVideo"
    # scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scenes = [0, 1, 2]
    pool = multiprocessing.Pool(processes=15)

    __refresh_dir(DATASETS_PATH / DATASET_NAME / "ablation")
    base_params = GlobalParams(
                                cov_drift_lin_vel=0.1,
                                cov_drift_ang_vel=1,
                                outlier_rejection_treshold=0.15,
                                t_validity_treshold=0.000005,
                                R_validity_treshold=0.00075,
                                max_derivative_order=1,
                                reject_overlaps=0.05)
    for cdlv in [1]:
        for cdav in [1]:
            for ort in [1]:
                forked_params = copy.deepcopy(base_params)
                forked_params.cov_drift_lin_vel = base_params.cov_drift_lin_vel * cdlv
                forked_params.cov_drift_ang_vel = base_params.cov_drift_ang_vel * cdav
                forked_params.outlier_rejection_treshold = base_params.outlier_rejection_treshold * ort
                # pool.apply_async(anotate_dataset, args=(DATASETS_PATH, DATASET_NAME, scenes, forked_params, dataset_type))
                anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, forked_params)
    # pool.close()
    # pool.join()

    # refined_scene = load_pickle(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'frames_refined_prediction.p')
    # scene_gt = utils.load_scene_gt(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'scene_gt.json', list(utils.HOPE_OBJECT_NAMES.values()))
    # scene_camera = load_scene_camera(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}" / "scene_camera.json")
    # animate_refinement(refined_scene, scene_gt, scene_camera)

    print(f"elapsed time: {time.time() - start_time:.2f}s")
    print(f"elapsed time: {utils.format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()
