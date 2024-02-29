import pickle
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple
import json
from collections import defaultdict

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
OBJ_IDS = {"AlphabetSoup":1,
           "BBQSauce":2,
           "Butter":3,
           "Cherries":4,
           "ChocolatePudding":5,
           "Cookies":6,
           "Corn":7,
           "CreamCheese":8,
           "GranolaBars":9,
           "GreenBeans":10,
           "Ketchup":11,
           "MacaroniAndCheese":12,
           "Mayo":13,
           "Milk":14,
           "Mushrooms":15,
           "Mustard":16,
           "OrangeJuice":17,
           "Parmesan":18,
           "Peaches":19,
           "PeasAndCarrots":20,
           "Pineapple":21,
           "Popcorn":22,
           "Raisins":23,
           "SaladDressing":24,
           "Spaghetti":25,
           "TomatoSauce":26,
           "Tuna":27,
           "Yogurt":28}

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_gt(path, label_list = None):
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
            entry[label_list[obj_id-1]].append(T_cm)
        parsed_data.append(entry)
    return parsed_data


def calculate_D(gt_T_co_s, est_T_co_s):
    """
    Calculates a 2d matrix containing distances between each new and old object estimates
    """
    D = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    A = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    for i in range(len(gt_T_co_s)):
        for j in range(len(est_T_co_s)):
            T_oo:pin.SE3 = pin.SE3(gt_T_co_s[i]).inverse()*pin.SE3(est_T_co_s[j])
            w = pin.log3(T_oo.rotation)
            t = T_oo.translation
            D[i, j] = np.linalg.norm(t)
            A[i, j] = np.linalg.norm(w)*180/np.pi
    return D, A

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

def plot_expected(scene_gt, scene_refined_prediction, axis, title=""):
    obj_to_y_map = {}
    obj_to_color_map = {}
    # colors = np.random.rand(100, 3)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    density = 20
    for a, obj_name in enumerate(scene_gt[0]):
        for idx, inst in enumerate(scene_gt[0][obj_name]):
            obj_to_y_map[f"{obj_name}{idx}"] = len(obj_to_y_map)
            if f"{obj_name}" not in obj_to_color_map:
                obj_to_color_map[f"{obj_name}"] = colors[len(obj_to_color_map)]

    y_gt = np.full((len(scene_gt)), None)
    for frame in range(0, len(scene_gt), density):
        for obj_name in scene_gt[frame]:
            for idx, inst in enumerate(scene_gt[frame][obj_name]):
                y = np.linalg.norm(pin.log6(pin.SE3(scene_gt[frame][obj_name][idx])).vector)
                axis.plot([frame], [y], 'o', color=obj_to_color_map[f"{obj_name}"], markersize=12, markeredgewidth=2, mfc='none')
                # y_gt[y, frame] = y
    id_to_track_map = {}
    track_to_obj_map = {}
    tracks =  np.full((0, len(scene_gt)//density + 1), None)
    for frame in range(0, len(scene_gt), density):
        for obj_name in scene_refined_prediction[frame]:
            for idx, inst in enumerate(scene_refined_prediction[frame][obj_name]):
                if inst["id"] not in id_to_track_map:
                    id_to_track_map[inst["id"]] = len(id_to_track_map)
                    tracks = np.concatenate((tracks, np.full((1, len(scene_gt)//density + 1), None)), axis=0)
                if f"{obj_name}" not in obj_to_color_map:
                    obj_to_color_map[f"{obj_name}"] = colors[len(obj_to_color_map)]
                track_to_obj_map[id_to_track_map[inst["id"]]] = f"{obj_name}"
                y = np.linalg.norm(pin.log6(pin.SE3(inst["T_co"])).vector)
                tracks[id_to_track_map[inst["id"]], frame//density] = y
    for i, track in enumerate(tracks):
        axis.plot(np.arange(0, tracks.shape[1])*density, track, '-o', color=obj_to_color_map[track_to_obj_map[i]])#, markersize=5, markeredgewidth=2)
    axis.set_title(title)
    # axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("")
    axis.set_xlim(-10, len(scene_gt))
    axis.grid()
    bbox = dict(boxstyle='round', facecolor='grey', alpha=0.1)
    for i, obj_name in enumerate(obj_to_color_map):
        plt.text(0.83, 0.90 - i/15, f"o {obj_name}",
                 fontsize=16, transform=plt.gcf().transFigure, bbox=bbox, color=obj_to_color_map[obj_name])
def plot_expected_cosypose(scene_gt, scene_refined_prediction, axis, title=""):
    obj_to_y_map = {}
    obj_to_color_map = {}
    # colors = np.random.rand(100, 3)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    density = 20
    for a, obj_name in enumerate(scene_gt[0]):
        for idx, inst in enumerate(scene_gt[0][obj_name]):
            obj_to_y_map[f"{obj_name}{idx}"] = len(obj_to_y_map)
            if f"{obj_name}" not in obj_to_color_map:
                obj_to_color_map[f"{obj_name}"] = colors[len(obj_to_color_map)]

    y_gt = np.full((len(scene_gt)), None)
    for frame in range(0, len(scene_gt), density):
        for obj_name in scene_gt[frame]:
            for idx, inst in enumerate(scene_gt[frame][obj_name]):
                y = np.linalg.norm(pin.log6(pin.SE3(scene_gt[frame][obj_name][idx])).vector)
                axis.plot([frame], [y], 'o', color=obj_to_color_map[f"{obj_name}"], markersize=12, markeredgewidth=2, mfc='none')
                # y_gt[y, frame] = y
    for frame in range(0, len(scene_gt), density):
        for obj_name in scene_refined_prediction[frame]:
            for idx, inst in enumerate(scene_refined_prediction[frame][obj_name]):
                if f"{obj_name}" not in obj_to_color_map:
                    obj_to_color_map[f"{obj_name}"] = colors[len(obj_to_color_map)]
                y = np.linalg.norm(pin.log6(pin.SE3(inst)).vector)
                axis.plot([frame], [y], 'o', color=obj_to_color_map[f"{obj_name}"], markersize=5, markeredgewidth=2)

    axis.set_title(title)
    # axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("")
    axis.set_xlim(-10, len(scene_gt))
    axis.grid()

def main():
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_PATH = DATASETS_PATH / "SynthStatic"
    # DATASET_PATH = DATASETS_PATH / "SynthDynamic"
    # DATASET_PATH = DATASETS_PATH / "hopeVideo"
    SCENES_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]
    num_of_plots = 1
    first_scene = 0
    figure, axis = plt.subplots(num_of_plots*2, 1)

    for i, dataset_name in enumerate(SCENES_NAMES[first_scene:min(first_scene+num_of_plots, len(SCENES_NAMES))]):
        dataset_path = DATASET_PATH / "test" / dataset_name
        scene_gt = load_scene_gt(dataset_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
        frames_prediction = load_data(dataset_path / "frames_prediction.p")
        frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")
        plot_expected(scene_gt, frames_refined_prediction, axis[i], title=f"{dataset_name} gtsam")
        plot_expected_cosypose(scene_gt, frames_prediction, axis[i+1], title=f"{dataset_name} CosyPose")

    plt.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()


if __name__ == "__main__":
    main()