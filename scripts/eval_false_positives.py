import pickle
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
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
        if (obj_name in frame_gt and len(frame_gt[obj_name]) > 0) and (obj_name in frame_est and len(frame_est[obj_name]) > 0):
            # count += max(0, (len(frame_gt[obj_name]) - len(frame[obj_name]))*false_negative)
            D, A = calculate_D(frame_gt[obj_name], frame_est[obj_name])
            if D.shape[0] == 0 or D.shape[1] == 0:
                continue
            D_bin = ((D < 0.1) & (A < 25)).astype(int)
            if false_negative == 1:
                D_bin = D_bin.T
            count += D_bin.shape[1] - np.linalg.matrix_rank(D_bin)
        elif (obj_name in frame_gt and len(frame_gt[obj_name]) > 0):
            count += max(0, false_negative*len(frame_gt[obj_name]))
        elif (obj_name in frame_est and len(frame_est[obj_name]) > 0):
            count += max(0, -false_negative * len(frame_est[obj_name]))
    return count

def calculate_D(gt_T_co_s, est_T_co_s):
    """
    Calculates a 2d matrix containing distances between each new and old object estimates
    """
    D = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    A = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    for i in range(len(gt_T_co_s)):
        for j in range(len(est_T_co_s)):
            if isinstance(est_T_co_s[j], dict):
                T_oo:pin.SE3 = pin.SE3(gt_T_co_s[i]).inverse()*pin.SE3(est_T_co_s[j]["T_co"])
            else:
                T_oo: pin.SE3 = pin.SE3(gt_T_co_s[i]).inverse() * pin.SE3(est_T_co_s[j])
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
    # DATASET_PATH = DATASETS_PATH / "SynthDynamic"
    # DATASET_PATH = DATASETS_PATH / "hopeVideo"
    SCENES_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]
    num_of_plots = 10
    figure, axis = plt.subplots(num_of_plots, 2)
    all_false_negative_frames = np.array([])
    all_false_negative_frames_refined = np.array([])
    all_false_positive_frames_refined = np.array([])
    all_false_positive_frames = np.array([])

    for i, dataset_name in enumerate(SCENES_NAMES[:num_of_plots]):
        dataset_path = DATASET_PATH / "test" / dataset_name
        scene_gt = load_scene_gt(dataset_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
        frames_prediction = load_data(dataset_path / "frames_prediction.p")
        frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")

        false_negative_frames = get_differences(scene_gt, frames_prediction, 1)
        false_negative_frames_refined = get_differences(scene_gt, frames_refined_prediction, 1)

        false_positive_frames_refined = get_differences(scene_gt, frames_refined_prediction, -1)
        false_positive_frames = get_differences(scene_gt, frames_prediction, -1)
        # print(f"{dataset_name}:")
        # print(f"CosyPose: avg. false negatives:{np.mean(false_negative_frames):.2f}, false positives:{np.mean(false_positive_frames):.2f}")
        # print(f"Gtsam avg. false negatives:{np.mean(false_negative_frames_refined):.2f}, false positives:{np.mean(false_positive_frames_refined):.2f}")
        # print("")
        all_false_negative_frames = np.concatenate((all_false_negative_frames, false_negative_frames), axis=0)
        all_false_negative_frames_refined = np.concatenate((all_false_negative_frames_refined, false_negative_frames_refined), axis=0)
        all_false_positive_frames_refined = np.concatenate((all_false_positive_frames_refined, false_positive_frames_refined), axis=0)
        all_false_positive_frames = np.concatenate((all_false_positive_frames, false_positive_frames), axis=0)
    print(f"CosyPose: avg. false negatives:{np.mean(all_false_negative_frames):.2f}, false positives:{np.mean(all_false_positive_frames):.2f}")
    print(f"Gtsam avg. false negatives:{np.mean(all_false_negative_frames_refined):.2f}, false positives:{np.mean(all_false_positive_frames_refined):.2f}")
    print(f"Gtsam avg. improvement:{100*np.mean(all_false_negative_frames_refined)/np.mean(all_false_negative_frames):.2f}%,"
          f"  {100*np.mean(all_false_positive_frames_refined)/np.mean(all_false_positive_frames):.2f}%")
    #     plot_differences([false_negative_frames, false_negative_frames_refined], axis[i, 0], title="false_negatives")
    #     plot_differences([false_positive_frames, false_positive_frames_refined], axis[i, 1], title="false_positives")
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    # plt.show()

# if t_det > 0.0000005 or R_det > 0.000005:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:9.20, false positives:0.58

# if t_det > 0.000001 or R_det > 0.00001:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:7.70, false positives:0.89

# if t_det > 0.000005 or R_det > 0.00005:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:4.82, false positives:2.02

# if t_det > 0.000005 or R_det > 0.00001:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:7.70, false positives:0.89

# if t_det > 0.000001 or R_det > 0.00003:
# time_elapsed = 0.0000001
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:5.51, false positives:1.56

# if t_det > 0.000001 or R_det > 0.00003:
# time_elapsed = 0.00001
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:6.21, false positives:1.09

# if t_det > 0.000001 or R_det > 0.00003:
# time_elapsed = 0.0001
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:9.09, false positives:0.42

# if t_det > 0.000003 or R_det > 0.00006:
# time_elapsed = 0.0001
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:6.84, false positives:0.75

# if t_det > 0.000006 or R_det > 0.00012:
# time_elapsed = 0.0001
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:6.02, false positives:1.21

# if t_det > 0.000006 or R_det > 0.00012:
# time_elapsed = 0.00000000001
# if minimum < 20:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:4.08, false positives:3.04

# if t_det > 0.000006 or R_det > 0.00012:
# time_elapsed = 0.00000000001
# if minimum < 10:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:4.09, false positives:3.04

# if t_det > 0.000006 or R_det > 0.00012:
# time_elapsed = 0.00000000001
# if minimum < 40:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:4.32, false positives:2.70

# if t_det > 0.000003 or R_det > 0.00006:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%2 == 0:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:6.02, false positives:1.36

# if t_det > 0.000003 or R_det > 0.00006:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:8.26, false positives:0.76

# if t_det > 0.000003 or R_det > 0.00003:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:9.40, false positives:0.46

# if t_det > 0.000006 or R_det > 0.00006:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# CosyPose: avg. false negatives:6.39, false positives:1.38
# Gtsam avg. false negatives:8.26, false positives:0.76

# if t_det > 0.00001 or R_det > 0.00006:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# self.MAX_AGE = 30
# CosyPose: avg. false negatives:7.82, false positives:2.81
# Gtsam avg. false negatives:8.56, false positives:1.07
# Gtsam avg. improvement:109.59%,  38.25%

# if t_det > 0.00001 or R_det > 0.00003:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# self.MAX_AGE = 30
# CosyPose: avg. false negatives:7.82, false positives:2.81
# Gtsam avg. false negatives:9.49, false positives:0.55
# Gtsam avg. improvement:121.47%,  19.74%

# if t_det > 0.00001 or R_det > 0.0001:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# self.MAX_AGE = 30
# CosyPose: avg. false negatives:7.82, false positives:2.81
# Gtsam avg. false negatives:7.71, false positives:1.70
# Gtsam avg. improvement:98.62%,  60.60%

# if t_det > 0.00001 or R_det > 0.00015:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%4 == 0:
# self.MAX_AGE = 30
# CosyPose: avg. false negatives:7.82, false positives:2.81
# Gtsam avg. false negatives:7.16, false positives:2.28
# Gtsam avg. improvement:91.64%,  81.36%

# if t_det > 0.00001 or R_det > 0.00010:
# time_elapsed = 0.00000000001
# if minimum < 20:
# if i%1 == 0:
# self.MAX_AGE = 30

# SynthStatic
# if i%4 == 0:
# if t_det > 0.000003 or R_det > 0.00005:
# CosyPose: avg. false negatives:1.83, false positives:2.17
# Gtsam avg. false negatives:0.88, false positives:2.03
# Gtsam avg. improvement:47.88%,  93.53%

# SynthStatic
# if i%4 == 0:
# if t_det > 0.000001 or R_det > 0.00002:
# CosyPose: avg. false negatives:1.83, false positives:2.17
# Gtsam avg. false negatives:1.70, false positives:1.46
# Gtsam avg. improvement:92.97%,  67.19%


if __name__ == "__main__":
    main()