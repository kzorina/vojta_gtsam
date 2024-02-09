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

def count_frame_difference(frame_gt, frame, false_negative):  # if false_positive=1 => counts number of false positives, if false_positive=-1 => counts false negatives
    count = 0
    for obj_name in set(frame_gt.keys())|set(frame.keys()):
        if (obj_name in frame_gt) and (obj_name in frame):
            count += max(0, (len(frame_gt[obj_name]) - len(frame[obj_name]))*false_negative)
        elif (obj_name in frame_gt):
            count += max(0, false_negative*len(frame_gt[obj_name]))
        elif (obj_name in frame):
            count += max(0, -false_negative*len(frame[obj_name]))
    return count

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
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo")
    DATASET_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]
    num_of_plots = 3
    figure, axis = plt.subplots(num_of_plots, 2)
    for i, dataset_name in enumerate(DATASET_NAMES[:num_of_plots]):
        dataset_path = DATASETS_PATH / "test" / dataset_name
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