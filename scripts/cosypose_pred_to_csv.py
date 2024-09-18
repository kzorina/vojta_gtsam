from pathlib import Path
import time
import numpy as np
import json
import pickle
import os
import copy
import bop_tools
import shutil

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_camera(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for k, v in data.items():
        entry = {}
        entry["cam_K"] = np.array(v["cam_K"]).reshape((3, 3))
        T_cw = np.zeros((4, 4))
        T_cw[:3, :3] = np.array(v["cam_R_w2c"]).reshape((3, 3))
        T_cw[:3, 3] = np.array(v["cam_t_w2c"])/1000
        T_cw[3, 3] = 1
        entry["T_cw"] = T_cw
        parsed_data.append(entry)
    return parsed_data

def cosypose_pred(DATASET_PATH, DATASET_NAME):
    dataset_path = DATASET_PATH / "test" / f"{DATASET_NAME:06}"
    frames_gt = load_scene_camera(dataset_path / "scene_camera.json")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    px_counts = load_data(dataset_path / "frames_px_counts.p")
    images = sorted(os.listdir(dataset_path / "rgb"))
    inference = []
    for i, img_name in enumerate(images):
        poses_dict = copy.deepcopy(frames_prediction[i])
        # T_wc_0 = np.linalg.inv(frames_gt[i_0]['T_cw'])
        # T_cw = frames_gt[i]['T_cw']
        # for key in frames_prediction[i_0]:
        #     for obj_idx in range(len(frames_prediction[i_0][key])):
        #         T_co_0 = frames_prediction[i_0][key][obj_idx]
        #         T_co = T_cw @ T_wc_0 @ T_co_0
        #         poses[key][obj_idx] = T_co
        #         print('')
        #         # poses[key]
        # print('')
        inference.append(poses_dict)
        print(f"\r({(i + 1)}/{len(images)})", end='')

    print("")
    # with open(dataset_path / f'frames_refined_prediction.p', 'wb') as file:
    #     pickle.dump(inference, file)
    # frames: [{"object_name": [T_co, T_co, T_co...]}]}
    img_ids = [int(name.split('.')[0]) for name in images]
    return inference, img_ids

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

def main():
    start_time = time.time()
    dataset_type = "ycbv"  # hope, ycbv

    DATASETS_PATH = Path("/home/kzorina/work/bop_datasets")
    # DATASET_NAME = "hopeVideo"
    DATASET_NAME = "ycbv_test_bop19"
    # DATASET_NAME = "SynthDynamic"
    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    __refresh_dir(DATASET_PATH / "ablation")
    datasets = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    # datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets = [0, 1, 2]
    # datasets = [0]
    # for mod in [2, 4, 6, 8, 10]:
    results = {}
    img_ids = {}
    for dataset_num in datasets:
        results[dataset_num], img_ids[dataset_num] = cosypose_pred(DATASET_PATH, dataset_num)
    bop_tools.export_bop(bop_tools.convert_frames_to_bop(results, dataset_type), DATASET_PATH / "ablation" / 'cosypose.csv')
    # merge_inferences(DATASET_PATH, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "frames_prediction.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()

if __name__ == "__main__":
    main()