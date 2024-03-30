from pathlib import Path
import json
import numpy as np
import pickle
import time
import utils
from utils import load_scene_camera, load_pickle, merge_T_cos_px_counts
from SamWrapper import SamWrapper
from GlobalParams import GlobalParams


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
        refined_scene.append(current_state)
        print(f"\r({(i + 1)}/{len(scene_camera)})", end='')
    return refined_scene

def refine_dataset(DATASETS_PATH, DATASET_NAME, scenes, params):
    for scene_num in scenes:
        scene_path = DATASETS_PATH/DATASET_NAME/"test"/ f"{scene_num:06}"
        scene_camera = load_scene_camera(scene_path / "scene_camera.json")
        frames_prediction = load_pickle(scene_path / "frames_prediction.p")
        px_counts = load_pickle(scene_path / "frames_px_counts.p")
        refined_scene = refine_scene(scene_camera, frames_prediction, px_counts, params)
        print("")

def main():
    start_time = time.time()
    dataset_type = "hope"
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "SynthDynamicOcclusion"
    # scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scenes = [0, 1, 2]
    params = GlobalParams()
    refine_dataset(DATASETS_PATH, DATASET_NAME, scenes, params)
    print(f"elapsed time: {time.time() - start_time:.2f}s")
    print(f"elapsed time: {utils.format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()
