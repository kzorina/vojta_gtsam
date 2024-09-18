import bop_tools
from pathlib import Path
import json
import numpy as np
import os
from collections import defaultdict

def load_scene_gt(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = defaultdict(lambda : [])
        frame = i+1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            entry[obj_id].append(T_cm)
        parsed_data.append(entry)
    return parsed_data

def insert_invalid_detections(results, ammount):
    T_cm = np.eye(4)
    T_cm[3, 3] = -10000
    obj_id = -1
    for idx in range(1, 20):
        if idx not in results[0].keys():
            obj_id = idx
            break
    obj_id = list(results[0].keys())[0]
    for frame in range(len(results)):
        T_cm = results[frame][obj_id][0]
        for i in range(ammount):
            results[frame][obj_id].append(T_cm)

def merge_inferences(DATASET_PATH, dataset_name="ycbv"):
    all_results = {}
    for scene in os.listdir(DATASET_PATH / "test"):
        scene_path = DATASET_PATH / "test" / scene
        results = load_scene_gt(scene_path / "scene_gt.json")
        insert_invalid_detections(results, 5)
        all_results[int(scene)] = results
    bop_tools.export_bop(bop_tools.convert_frames_to_bop(all_results, dataset_name, translate_obj_ids=False), DATASET_PATH / "augmentedGroundTruth_SynthStaticDummy-test_3.csv")

def main():
    DATASETS_PATH = Path("/home/kzorina/work/bop_datasets")
    DATASET_NAME = "hopeVideo"
    merge_inferences(DATASETS_PATH/DATASET_NAME, "hope")

if __name__ == "__main__":
    main()


    "/home/kzorina/work/bop_datasets/hopeVideo/augmentedGroundTruth_SynthStaticDummy-test_3.csv"