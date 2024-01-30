import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_pos_error(Tco_gt, Tco_estimate):
    Too = np.linalg.inv(Tco_gt)@Tco_estimate
    error = np.linalg.norm(Too[:3, 3])
    return error

def get_rot_error(Tco_gt, Tco_estimate):
    Too = np.linalg.inv(Tco_gt)@Tco_estimate
    rot = Rotation.from_matrix(Too[:3, :3])
    angle = np.linalg.norm(rot.as_rotvec())
    return angle

def plot_error_vectors(error_vectors, axis:plt.axis, title = ""):
    axis.set_title(title)
    colors = ["tab:red", "tab:green"]
    # colors = ["firebrick", "gold"]
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector*1000, 'r-o', color=colors[i])
    axis.legend(["cosypose", "gtsam"])
    axis.set_xlabel("frame")
    axis.set_ylabel("error norm[mm]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.set_ylim(bottom=0)
    axis.grid()
    # plt.savefig(f'{title}.png')

def plot_error_vectors_rot(error_vectors, axis:plt.axis, title = ""):
    axis.set_title(title)
    colors = ["tab:red", "tab:green"]
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector * 180 / np.pi, 'r-o', color=colors[i])
    axis.legend(["cosypose", "gtsam"])
    axis.set_xlabel("frame")
    axis.set_ylabel("error angle[mm]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.grid()

def get_object_error_vector(frames_gt, frames_prediction, obj_name):
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
    return pos_error_vectors, rot_error_vectors


def plot_results(object_names, frames_gt, frames_prediction, frames_refined_prediction):
    figure, axis = plt.subplots(len(object_names), 2)
    for i, name in enumerate(object_names):
        pos_errors, rot_errors = get_object_error_vector(frames_gt, frames_prediction, name)
        ref_pos_errors, ref_rot_errors = get_object_error_vector(frames_gt, frames_refined_prediction, name)
        plot_error_vectors(np.vstack((pos_errors, ref_pos_errors)), axis[i, 0], f"{name} translation error")
        plot_error_vectors_rot(np.vstack((rot_errors, ref_rot_errors)), axis[i, 1], f"{name} rot error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.45)
    plt.show()
def main():
    dataset_name = "crackers_new"
    # dataset_name = "crackers_duplicates"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    frames_gt = load_data(dataset_path/"frames_gt.p")
    # frames_prediction = load_data(dataset_path/"frames_prediction1.p")
    frames_prediction = load_data(dataset_path / "frames_prediction1.p")
    frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")
    objects_to_plot = ["02_cracker_box", "03_sugar_box", "07_pudding_box", "12_bleach_cleanser"]
    plot_results(objects_to_plot, frames_gt, frames_prediction, frames_refined_prediction)
    # figure, axis = plt.subplots(4, 2)
    # pos_errors, rot_errors = get_object_error_vector(frames_gt, frames_prediction, "02_cracker_box")
    # ref_pos_errors, ref_rot_errors = get_object_error_vector(frames_gt, frames_refined_prediction, "02_cracker_box")
    # plot_error_vectors(np.vstack((pos_errors, ref_pos_errors)), axis[0, 0], "02_cracker_box_error")
    # plot_error_vectors_rot(np.vstack((rot_errors, ref_rot_errors)), axis[0, 1], "02_cracker_box_error")


pass

if __name__ == "__main__":
    main()