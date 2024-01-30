import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_error_vector(Tco_gt, Tco_estimate):
    Too = np.linalg.inv(Tco_gt)@Tco_estimate
    error_vector = np.zeros(4)
    error_vector[:3] = Too[:3, 3]
    rot = Rotation.from_matrix(Too[:3, :3])
    angle = np.linalg.norm(rot.as_rotvec())
    error_vector[3] = angle
    return error_vector

def plot_error_vectors(error_vectors, axis:plt.axis, title = ""):
    axis.set_title(title)
    for i, error_vector in enumerate(error_vectors):
        axis.plot([i], error_vector[0]*1000, 'o', color="red")
        axis.plot([i], error_vector[1]*1000, 'o', color="green")
        axis.plot([i], error_vector[2]*1000, 'o', color="blue")
        axis.plot([i], error_vector[3]*180/np.pi, 'o', color="purple")

    axis.legend(["x", "y", "z", "rot"])
    axis.set_xlabel("frame")
    axis.set_ylabel("xyz[mm], rot[deg]")
    axis.set_xlim(-1, len(error_vectors))
    axis.grid()
    # plt.savefig(f'{title}.png')

def get_object_error_vector(frames_gt, frames_prediction, obj_name):
    error_vectors = []
    for frame in range(len(frames_gt)):
        if (obj_name in frames_prediction[frame]):
            T_co_gt = frames_gt[frame][obj_name]
            T_co_estimate = frames_prediction[frame][obj_name]
            error_vectors.append(get_error_vector(T_co_gt, T_co_estimate))
        else:
            error_vectors.append(np.zeros(4))
    return error_vectors

def get_absolute_object_error_vector(frames_gt, frames_prediction, obj_name):
    error_vectors = []
    for frame in range(len(frames_gt)):
        T_co_gt = frames_gt[frame][obj_name]
        T_co_estimate = frames_prediction[frame][obj_name]
        error_vectors.append(get_error_vector(T_co_gt, T_co_estimate))
    return error_vectors

def main():
    dataset_name = "crackers_new"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    frames_gt = load_data(dataset_path/"frames_gt.p")
    # frames_prediction = load_data(dataset_path/"frames_prediction1.p")
    frames_prediction = load_data(dataset_path / "frames_refined_prediction.p")

    figure, axis = plt.subplots(4)
    error_vectors = get_object_error_vector(frames_gt, frames_prediction, "02_cracker_box")
    plot_error_vectors(error_vectors, axis[0], "02_cracker_box_error")
    error_vectors = get_object_error_vector(frames_gt, frames_prediction, "03_sugar_box")
    plot_error_vectors(error_vectors, axis[1], "03_sugar_box_error")
    error_vectors = get_object_error_vector(frames_gt, frames_prediction, "07_pudding_box")
    plot_error_vectors(error_vectors, axis[2], "07_pudding_box_error")
    error_vectors = get_object_error_vector(frames_gt, frames_prediction, "12_bleach_cleanser")
    plot_error_vectors(error_vectors, axis[3], "12_bleach_cleanser_error")
    plt.show()
pass

if __name__ == "__main__":
    main()