import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pinocchio as pin
def load_scene_camera(path):  # used for ycbv datasets
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

def load_scene_gt(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        frame = i+1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            entry[obj_id] = T_cm
        parsed_data.append(entry)
    return parsed_data

def plot_error_vectors(error_vectors, axis:plt.axis, title = "", legend = ("X", "Y", "Z"), colors = ("tab:red", "tab:green", "tab:blue")):
    axis.set_title(title)
    for i, error_vector in enumerate(error_vectors):
        axis.plot(np.arange(0, error_vector.shape[0]), error_vector*1000, '-', color=colors[i])
    axis.legend(legend)
    axis.set_xlabel("frame")
    axis.set_ylabel("error norm[mm]")
    axis.set_xlim(-1, len(error_vectors[0]))
    axis.set_ylim()
    axis.grid()

def plot_errors(errors):

    figure, axis = plt.subplots(len(errors))
    for i, obj_id in enumerate(errors):
        plot_error_vectors([errors[obj_id]], axis[i], f"obj_id:{obj_id}", legend=("translation_error"))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.8)
    plt.show()
def main():
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hope_video")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    # DATASET_NAMES = ["000048", "000049", "000050", "000051", "000052", "000053", "000054", "000055", "000056", "000057", "000058", "000059"]
    DATASET_NAMES = ["000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008"]

    for dataset_name in DATASET_NAMES[0:1]:
        scene_camera = load_scene_camera(DATASETS_PATH/"test"/dataset_name/"scene_camera.json")
        scene_gt = load_scene_gt(DATASETS_PATH/"test"/dataset_name/"scene_gt.json")
        T_wc_0 = pin.SE3(scene_camera[0]["T_cw"]).inverse()
        translation_errors = {}
        obj_ids = list(scene_gt[0].keys())[:5]
        # for obj_id in scene_gt[0]:
        for obj_id in obj_ids:
            translation_errors[obj_id] = np.zeros((len(scene_camera)))
        for frame in range(len(scene_camera)):
            T_wc = pin.SE3(scene_camera[frame]["T_cw"]).inverse()

            # for obj_id in scene_gt[0]:
            for obj_id in obj_ids:
                T_cm_0 = pin.SE3(scene_gt[0][obj_id])
                T_wm_0: pin.SE3 = T_wc_0 * T_cm_0
                T_mw_0 = T_wm_0.inverse()
                T_cm = pin.SE3(scene_gt[frame][obj_id])
                T_wm:pin.SE3 =  T_wc * T_cm
                T_mm = T_mw_0 * T_wm
                translation_errors[obj_id][frame] = np.linalg.norm(T_mm.translation)
        plot_errors(translation_errors)
if __name__ == "__main__":
    main()