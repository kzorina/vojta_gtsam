from pathlib import Path
import time
import numpy as np
import json
import pickle
import os
import copy
import pinocchio as pin
from collections import defaultdict
import shutil
import bop_tools

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_camera(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        entry["cam_K"] = np.array(data[str(i+1)]["cam_K"]).reshape((3, 3))
        T_cw = np.zeros((4, 4))
        T_cw[:3, :3] = np.array(data[str(i+1)]["cam_R_w2c"]).reshape((3, 3))
        T_cw[:3, 3] = np.array(data[str(i+1)]["cam_t_w2c"])/1000
        T_cw[3, 3] = 1
        entry["T_cw"] = T_cw
        parsed_data.append(entry)
    return parsed_data

def calculate_D(last_poses, new_poses):
    if len(last_poses) == 0:
        return None
    D = np.ndarray((len(last_poses), (len(new_poses))))
    for i in range(len(new_poses)):
        for j in range(len(last_poses)):
            last_T_wo = last_poses[j]
            new_T_wo = new_poses[i]
            T_oo:pin.SE3 = pin.SE3(new_T_wo).inverse()*pin.SE3(last_T_wo)
            rot_dist = np.linalg.norm(pin.log3(T_oo.rotation))/20
            trans_dist = np.linalg.norm(T_oo.translation)
            D[j, i] = rot_dist + trans_dist
    return D

def determine_assignment(D):
    assignment = []
    for i in range(D.shape[1]):
        argmin = np.argmin(D[:,i])
        if argmin < 0.2:
            # minimum = D[:,i][argmin]
            D[argmin, :] = np.full((D.shape[1]), np.inf)
            assignment.append(argmin)
        else:
            assignment.append(-1)
    return assignment


def estimate_velocities(T_wos, last_T_wos, time_stamp, last_time_stamp):
    velocities = {}
    elapsed_time = time_stamp - last_time_stamp
    for obj_label in T_wos:
        obj_velocities = []
        D = calculate_D(last_T_wos[obj_label], T_wos[obj_label])
        if D is None:
            assignment = [-1]*len(T_wos[obj_label])
        else:
            assignment = determine_assignment(D)
        for i in range(len(T_wos[obj_label])):
            T_wo = pin.SE3(T_wos[obj_label][i])
            j = assignment[i]
            if j == -1:
                obj_velocities.append(np.zeros(6))
                continue
            last_T_wo = pin.SE3(last_T_wos[obj_label][j])
            o = pin.log3((T_wo * last_T_wo.inverse()).rotation)
            p = T_wo.translation - last_T_wo.translation
            if elapsed_time < 0.000001:
                obj_velocities.append(np.zeros(6))
            else:
                obj_velocities.append(np.concatenate([o, p])/elapsed_time)
        velocities[obj_label] = obj_velocities
    return velocities
def plus_so3r3_global(T: pin.SE3, nu: np.ndarray, dt: float):
    o, v = nu[:3], nu[3:]
    R_new = pin.exp3(o * dt) @ T.rotation
    t_new = T.translation + v * dt
    # return gtsam.Pose3(gtsam.Rot3.Expmap(do*dt), dv*dt) * T
    return pin.SE3(R_new, t_new)

def T_cos_to_T_wos(predictions, T_wc:pin.SE3):
    ret = defaultdict(list)
    for obj_label in predictions:
        for obj_idx in range(len(predictions[obj_label])):
            T_co = pin.SE3(predictions[obj_label][obj_idx])
            T_wo = T_wc * T_co
            ret[obj_label].append(T_wo)
    return ret

def cosypose_mod(DATASET_PATH, DATASET_NAME, mod):
    dataset_path = DATASET_PATH / "test" / f"{DATASET_NAME:06}"
    frames_gt = load_scene_camera(dataset_path / "scene_camera.json")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    px_counts = load_data(dataset_path / "frames_px_counts.p")
    images = sorted(os.listdir(dataset_path / "rgb"))
    inference = []
    for i, img_name in enumerate(images):
        i_0 = i - i % mod
        last_i_0 = max(0, i_0 - mod)
        poses = copy.deepcopy(frames_prediction[i_0])
        T_cos_0 = frames_prediction[i_0]
        T_cw_0 = pin.SE3(frames_gt[i_0]['T_cw'])
        T_wos_0 = T_cos_to_T_wos(T_cos_0, T_cw_0.inverse())
        time_stamp = i_0/30

        last_T_cos_0 = frames_prediction[last_i_0]
        last_T_cw_0 = pin.SE3(frames_gt[last_i_0]['T_cw'])
        last_T_wos_0 = T_cos_to_T_wos(last_T_cos_0, last_T_cw_0.inverse())
        last_time_stamp = last_i_0/30

        velocities = estimate_velocities(T_wos_0, last_T_wos_0, time_stamp, last_time_stamp)
        T_cw = pin.SE3(frames_gt[i]['T_cw'])
        for key in T_wos_0:
            for obj_idx in range(len(T_wos_0[key])):
                T_wo_0 = T_wos_0[key][obj_idx]
                nu = velocities[key][obj_idx]
                T_wo = plus_so3r3_global(T_wo_0, nu, i/30 - i_0/30)
                # T_wo = T_wo_0
                T_co:pin.SE3 = T_cw * T_wo
                poses[key][obj_idx] = T_co.homogeneous
                print('')
                # poses[key]

        print('')
        inference.append(poses)
        print(f"\r({(i + 1)}/{len(images)})", end='')

    print("")
    with open(dataset_path / f'frames_refined_prediction.p', 'wb') as file:
        pickle.dump(inference, file)
    return inference

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
    dataset_type = "hope"

    DATASETS_PATH = Path("/home/kzorina/work/bop_datasets")
    # DATASET_NAME = "SynthDynamicOcclusion"
    # DATASET_NAME = "SynthDynamic"
    DATASET_NAME = "hopeVideo"
    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    __refresh_dir(DATASET_PATH / "ablation")
    # datasets = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets = [2]
    # datasets = [0]
    for mod in [4]:
        results = {}
        for dataset_num in datasets:
            results[dataset_num] = cosypose_mod(DATASET_PATH, dataset_num, mod=mod)
        output_name = f'mod_{DATASET_NAME}-test_{mod}_.csv'
        bop_tools.export_bop(bop_tools.convert_frames_to_bop(results, dataset_type), DATASET_PATH / "ablation" / output_name)
    # merge_inferences(DATASET_PATH, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "frames_prediction.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()

if __name__ == "__main__":
    main()