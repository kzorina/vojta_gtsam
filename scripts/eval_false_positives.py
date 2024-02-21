import pickle
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple
import json

HOPE_OBJECT_NAMES = {"obj_000001": "AlphabetSoup",
    "obj_000002": "BBQSauce",
    "obj_000003": "Butter",
    "obj_000004": "Cherries",
    "obj_000005": "ChocolatePudding",
    "obj_000006": "Cookies",
    "obj_000007": "Corn",
    "obj_000008": "CreamCheese",
    "obj_000009": "GranolaBars",
    "obj_000010": "GreenBeans",
    "obj_000011": "Ketchup",
    "obj_000012": "MacaroniAndCheese",
    "obj_000013": "Mayo",
    "obj_000014": "Milk",
    "obj_000015": "Mushrooms",
    "obj_000016": "Mustard",
    "obj_000017": "OrangeJuice",
    "obj_000018": "Parmesan",
    "obj_000019": "Peaches",
    "obj_000020": "PeasAndCarrots",
    "obj_000021": "Pineapple",
    "obj_000022": "Popcorn",
    "obj_000023": "Raisins",
    "obj_000024": "SaladDressing",
    "obj_000025": "Spaghetti",
    "obj_000026": "TomatoSauce",
    "obj_000027": "Tuna",
    "obj_000028": "Yogurt"}

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_gt(path, label_list = None):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        frame = i+1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            if label_list is not None:
                entry[label_list[obj_id-1]] = [T_cm]
            else:
                entry[obj_id] = [T_cm]
        parsed_data.append(entry)
    return parsed_data

def count_frame_difference_old(frame_gt, frame, false_negative):  # if false_positive=1 => counts number of false positives, if false_positive=-1 => counts false negatives
    count = 0
    for obj_name in set(frame_gt.keys())|set(frame.keys()):
        if (obj_name in frame_gt) and (obj_name in frame):
            count += max(0, (len(frame_gt[obj_name]) - len(frame[obj_name]))*false_negative)
        elif (obj_name in frame_gt):
            count += max(0, false_negative*len(frame_gt[obj_name]))
        elif (obj_name in frame):
            count += max(0, -false_negative*len(frame[obj_name]))
    return count

def count_frame_difference(frame_gt, frame_est, false_negative):  # if false_negative=1 => counts number of false negatives, if false_negative=-1 => counts false positives
    count = 0
    for obj_name in set(frame_gt.keys())|set(frame_est.keys()):
        if (obj_name in frame_gt) and (obj_name in frame_est):
            # count += max(0, (len(frame_gt[obj_name]) - len(frame[obj_name]))*false_negative)
            D = calculate_D(frame_gt[obj_name], frame_est[obj_name])
            if D.shape[0] == 0 or D.shape[1] == 0:
                continue
            D_bin = np.zeros_like(D)
            D_bin[D < 0.3] = 1
            if false_negative == 1:
                D_bin = D_bin.T
            a = D_bin.shape[1] - np.linalg.matrix_rank(D_bin)
            # if (D.shape[0] > 1 or D.shape[1] > 1) and false_negative == 1:
            #     print("")
            count += a
        elif (obj_name in frame_gt):
            count += max(0, false_negative*len(frame_gt[obj_name]))
        elif (obj_name in frame_est):
            count += max(0, -false_negative * len(frame_est[obj_name]))
    return count

def calculate_D(gt_T_co_s, est_T_co_s):
    """
    Calculates a 2d matrix containing distances between each new and old object estimates
    """
    D = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    for i in range(len(gt_T_co_s)):
        for j in range(len(est_T_co_s)):
            T_oo:pin.SE3 = pin.SE3(gt_T_co_s[i]).inverse()*pin.SE3(est_T_co_s[j])
            w = pin.log3(T_oo.rotation)
            t = T_oo.translation
            D[i, j] = np.linalg.norm(w) + np.linalg.norm(t)
    return D

def determine_assignment(frames_gt, frames_prediction, object_name):
    assignment = []
    for frame in range(len(frames_gt)):
        if object_name not in frames_prediction[frame]:
            assignment.append(None)
            continue
        num_of_objects = min(len(frames_gt[frame][object_name]), len(frames_prediction[frame][object_name]))
        D = calculate_D(frames_gt[frame][object_name], frames_prediction[frame][object_name][:num_of_objects])
        entry = []
        for i in range(D.shape[1]):
            argmin = np.argmin(D[:,i])
            # minimum = D[:,i][argmin]
            D[argmin, :] = np.full((D.shape[1]), np.inf)
            entry.append(argmin)
        assignment.append(entry)
    return assignment

def get_differences(frames_gt, frames, false_negative):
    diff = np.zeros((len(frames_gt)))
    for i in range(len(frames_gt)):
        diff[i] = count_frame_difference(frames_gt[i], frames[i], false_negative)
    return diff

def plot_differences(differences, axis, title, legend = ("cosypose", "gtsam")):
    colors =("tab:red", "tab:green")
    axis.set_title(title)
    for i, error_vector in enumerate(differences):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector, '-', color=colors[i])
    axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("")
    axis.set_xlim(-1, len(differences[0]))
    axis.grid()

def main():
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_PATH = DATASETS_PATH / "SynthStatic"
    SCENES_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]
    num_of_plots = 3
    figure, axis = plt.subplots(num_of_plots, 2)
    for i, dataset_name in enumerate(SCENES_NAMES[:num_of_plots]):
        dataset_path = DATASET_PATH / "test" / dataset_name
        scene_gt = load_scene_gt(dataset_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
        frames_prediction = load_data(dataset_path / "frames_prediction.p")
        frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")

        false_negative_frames = get_differences(scene_gt, frames_prediction, 1)
        false_negative_frames_refined = get_differences(scene_gt, frames_refined_prediction, 1)

        false_positive_frames_refined = get_differences(scene_gt, frames_refined_prediction, -1)
        false_positive_frames = get_differences(scene_gt, frames_prediction, -1)

        plot_differences([false_negative_frames, false_negative_frames_refined], axis[i, 0], title="false_negatives")
        plot_differences([false_positive_frames, false_positive_frames_refined], axis[i, 1], title="false_positives")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()

if __name__ == "__main__":
    main()