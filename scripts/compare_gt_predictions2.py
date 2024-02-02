import pickle
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_pos_error(Tco_gt: np.ndarray, Tco_estimate: np.ndarray) -> float:
    """
    :param Tco_gt: np.array shape(4, 4)
    :param Tco_estimate: np.array shape(4, 4)
    :return: int
    """
    Too = np.linalg.inv(Tco_gt)@Tco_estimate
    error = np.linalg.norm(Too[:3, 3])
    return error

def get_xyz_error(Tco_gt: np.ndarray, Tco_estimate: np.ndarray, coordinate : int=0) -> np.ndarray:
    """
    :param Tco_gt: np.array shape(4, 4)
    :param Tco_estimate: np.array shape(4, 4)
    :return: int
    """
    # Too = np.linalg.inv(Tco_gt)@Tco_estimate
    error = Tco_gt[:3, 3] - Tco_estimate[:3, 3]
    return error[coordinate]

def get_rot_error(Tco_gt: np.ndarray, Tco_estimate: np.ndarray) -> float:
    """
    :param Tco_gt: np.ndarray shape(4, 4)
    :param Tco_estimate: np.ndarray shape(4, 4)
    :return: int
    """
    Too = np.linalg.inv(Tco_gt)@Tco_estimate
    rot = Rotation.from_matrix(Too[:3, :3])
    angle = np.linalg.norm(rot.as_rotvec())
    return angle

def plot_error_vectors(error_vectors, axis:plt.axis, title = "", legend = ("X", "Y", "Z"), colors = ("tab:red", "tab:green", "tab:blue")):
    axis.set_title(title)
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector*1000, 'r-o', color=colors[i])
    axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("error norm[mm]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.set_ylim()
    axis.grid()

def plot_error_vectors_rot(error_vectors, axis:plt.axis, title = "", legend=tuple("rot"), colors =("tab:red", "tab:green")):
    axis.set_title(title)
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector * 180 / np.pi, 'r-o', color=colors[i])
    axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("error angle[deg]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.grid()

def get_object_error_vector(frames_gt: List[Dict], frames_prediction: List[Dict], obj_name: str) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    pos_error_vectors = []
    rot_error_vectors = []
    for frame in range(len(frames_gt)):
        if (obj_name in frames_prediction[frame]):
            T_co_gt = frames_gt[frame][obj_name]
            T_co_estimate = frames_prediction[frame][obj_name]
            pos_error_vectors.append(get_pos_error(T_co_gt, T_co_estimate))
            rot_error_vectors.append(get_rot_error(T_co_gt, T_co_estimate))
        else:
            pos_error_vectors.append(0)
            rot_error_vectors.append(0)
    return np.array(pos_error_vectors), np.array(rot_error_vectors)

def get_object_split_error_vector(frames_gt: List[Dict], frames_prediction: List[Dict], obj_name: str, obj_idx_est: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_error_vectors = []
    Y_error_vectors = []
    Z_error_vectors = []
    rot_error_vectors = []
    assignment = determine_assignment(frames_gt, frames_prediction, obj_name)
    for frame in range(len(frames_gt)):
        if ((obj_name in frames_prediction[frame]) and (len(frames_prediction[frame][obj_name]) > obj_idx_est)):
            T_co_gt = frames_gt[frame][obj_name][assignment[frame][obj_idx_est]]
            T_co_estimate = frames_prediction[frame][obj_name][obj_idx_est]
            X_error_vectors.append(get_xyz_error(T_co_gt, T_co_estimate, 0))
            Y_error_vectors.append(get_xyz_error(T_co_gt, T_co_estimate, 1))
            Z_error_vectors.append(get_xyz_error(T_co_gt, T_co_estimate, 2))
            rot_error_vectors.append(get_rot_error(T_co_gt, T_co_estimate))
        else:
            X_error_vectors.append(0.0)
            Y_error_vectors.append(0.0)
            Z_error_vectors.append(0.0)
            rot_error_vectors.append(0.0)
    return np.array(X_error_vectors), np.array(Y_error_vectors), np.array(Z_error_vectors), np.array(rot_error_vectors)



def calculate_D(gt_T_co_s, est_T_co_s):
    """
    Calculates a 2d matrix containing distances between each new and old object estimates
    """
    D = np.ndarray((len(gt_T_co_s), len(est_T_co_s)))
    for i in range(len(gt_T_co_s)):
        for j in range(len(est_T_co_s)):
            T_oo:pin.SE3 = pin.SE3(gt_T_co_s[i]).inverse()*pin.SE3(est_T_co_s[j])
            D[i, j] = np.linalg.norm(pin.log6(T_oo))
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

def plot_results(object_names: Dict, frames_gt, frames_prediction):
    figure, axis = plt.subplots(max(sum(object_names.values()), 2), 2)
    estimators = ["gtsam", "cosypose"]
    colors = ["tab:red", "tab:blue"]
    i = 0
    for name in object_names:
        for j in range(object_names[name]):  # number of instances of a single object
            pos_errors = []  # estimators x frames
            rot_errors = []
            for p in range(len(frames_prediction)):
                x_errors, y_errors, z_errors, a_errors = get_object_split_error_vector(frames_gt, frames_prediction[p], name, j)
                pos_errors.append((x_errors**2 + y_errors**2 + z_errors**2)**0.5)
                rot_errors.append(a_errors)
            plot_error_vectors(pos_errors, axis[i, 0], f"{name} translation error", estimators, colors)
            plot_error_vectors_rot(rot_errors, axis[i, 1], f"{name} rot error", estimators, colors)
            i += 1
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()

def plot_split_results(object_names: Dict, frames_gt, frames_prediction):
    figure, axis = plt.subplots(sum(object_names.values()), 2)
    for prediction in frames_prediction:
        i = 0
        for name in object_names:
            assignment = determine_assignment(frames_gt, prediction, name)
            for j in range(object_names[name]):  # number of instances of a single object
                x_errors, y_errors, z_errors, rot_errors = get_object_split_error_vector(frames_gt, prediction, name, j, assignment)
                plot_error_vectors([x_errors, y_errors, z_errors], axis[i, 0], f"{name} translation error")
                plot_error_vectors_rot([rot_errors], axis[i, 1], f"{name} rot error")
                i += 1
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()

def main():
    # dataset_name = "static_medium"
    # dataset_name = "static1"
    # dataset_name = "dynamic1"
    dataset_name = "crackers_duplicates"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    frames_gt = load_data(dataset_path/"frames_gt.p")
    # frames_prediction = load_data(dataset_path/"frames_prediction1.p")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")
    # objects_to_plot = {"02_cracker_box":1, "11_pitcher_base":1, "15_power_drill":1, "17_scissors":1}
    # objects_to_plot = {"01_master_chef_can":1, "03_sugar_box":1, "05_mustard_bottle":1, "12_bleach_cleanser":1}
    # objects_to_plot = {"02_cracker_box":1}
    objects_to_plot = {"02_cracker_box":3}
    plot_results(objects_to_plot, frames_gt, [frames_refined_prediction, frames_prediction])
pass

if __name__ == "__main__":
    main()