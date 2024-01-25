import pickle
import numpy as np
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

def plot_error_vectors(error_vectors, axis:plt.axis, title = ""):
    axis.set_title(title)
    colors = ["tab:red", "tab:green", "tab:blue"]
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector*1000, 'o', color=colors[i])
    axis.legend(["X", "Y", "Z"])
    axis.set_xlabel("frame")
    axis.set_ylabel("error norm[mm]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.set_ylim()
    axis.grid()

def plot_error_vectors_rot(error_vectors, axis:plt.axis, title = ""):
    axis.set_title(title)
    colors = ["tab:red", "tab:green"]
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector * 180 / np.pi, 'r-o', color=colors[i])
    axis.legend(["rot"])
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

def get_object_split_error_vector(frames_gt: List[Dict], frames_prediction: List[Dict], obj_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_error_vectors = []
    Y_error_vectors = []
    Z_error_vectors = []
    rot_error_vectors = []
    for frame in range(len(frames_gt)):
        if (obj_name in frames_prediction[frame]):
            T_co_gt = frames_gt[frame][obj_name]
            T_co_estimate = frames_prediction[frame][obj_name]
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

def plot_results(object_names, frames_gt, frames_prediction):
    figure, axis = plt.subplots(len(object_names), 2)
    for i, name in enumerate(object_names):
            pos_errors, rot_errors = get_object_error_vector(frames_gt, frames_prediction, name)
            plot_error_vectors([pos_errors], axis[i, 0], f"{name} translation error")
            plot_error_vectors_rot([rot_errors], axis[i, 1], f"{name} rot error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()

def plot_split_results(object_names, frames_gt, frames_prediction):
    figure, axis = plt.subplots(len(object_names), 2)
    for i, name in enumerate(object_names):
        for prediction in frames_prediction:
            x_errors, y_errors, z_errors, rot_errors = get_object_split_error_vector(frames_gt, prediction, name)
            plot_error_vectors([x_errors, y_errors, z_errors], axis[i, 0], f"{name} translation error")
            plot_error_vectors_rot([rot_errors], axis[i, 1], f"{name} rot error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()

def main():
    dataset_name = "crackers_new"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    frames_gt = load_data(dataset_path/"frames_gt.p")
    # frames_prediction = load_data(dataset_path/"frames_prediction.p")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")
    objects_to_plot = ["02_cracker_box", "03_sugar_box", "07_pudding_box", "12_bleach_cleanser"]
    plot_results(objects_to_plot, frames_gt, frames_prediction, frames_refined_prediction)


pass

if __name__ == "__main__":
    main()