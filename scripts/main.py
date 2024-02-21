from pathlib import Path
from Detection import Detection
import os
import cv2
# from SAM_incremental_fixed_lag_smoother import SAM
# from SAM_isam2 import SAM
from SAM_fifo import SAM
import pickle
from compare_gt_predictions2 import plot_split_results
import time
import json
import numpy as np
from bop_tools import convert_frames_to_bop, export_bop
import matplotlib.pyplot as plt
import pinocchio as pin
import gtsam

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_time_delays(time_delays, landmark_counts):
    figure, axis = plt.subplots(1)
    colors =("tab:red", "tab:green")
    # axis[0].set_title("time delays")
    axis.plot(landmark_counts, time_delays, '-', color=colors[0])
    axis.legend("time delays")
    axis.set_xlabel("number of landmarks")
    axis.set_ylabel("time delays")
    axis.set_xlim(-1, landmark_counts[-1])
    axis.grid()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.4, hspace=0.45)
    bbox = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    plt.text(0.3, 0.10, f"frames: {len(time_delays)}\ntotal time: {time_delays.sum():.2f}s\nfps: {len(time_delays)/time_delays.sum():.2f}",
             fontsize=16, transform=plt.gcf().transFigure, bbox=bbox)
    plt.show()

def plot_estimate_progress(estimate_progress):
    figure, axis = plt.subplots(5, 2)
    seen_objects = list(estimate_progress[-1].keys())
    objects_to_plot = [(seen_objects[0], 0), (seen_objects[0], 1), (seen_objects[0], 2),
                       (seen_objects[1], 0), (seen_objects[1], 1)]
    for axis_id, (obj_name, i) in enumerate(objects_to_plot):
        t_cov_sizes = np.full((len(estimate_progress)), np.nan)
        R_cov_sizes = np.full((len(estimate_progress)), np.nan)
        for frame, objects in enumerate(estimate_progress):
            if obj_name in objects:
                if len(objects[obj_name]) > i:
                    Q = objects[obj_name][i]["Q"]
                    t_cov_sizes[frame] = np.linalg.det(Q[3:6, 3:6])**0.5
                    R_cov_sizes[frame] = np.linalg.det(Q[:3, :3])**0.5
        axis[axis_id, 0].plot(np.arange(0, t_cov_sizes.shape[0]), t_cov_sizes, '-')
        axis[axis_id, 1].plot(np.arange(0, R_cov_sizes.shape[0]), R_cov_sizes, '-')
        axis[axis_id, 0].grid()
        axis[axis_id, 1].grid()
        axis[axis_id, 0].set_xlabel("frame")
        axis[axis_id, 0].set_ylabel("sqrt_trans_cov_det")
        axis[axis_id, 1].set_xlabel("frame")
        axis[axis_id, 1].set_ylabel("sqrt_rot_cov_det")
        axis[axis_id, 0].set_title(f"{obj_name}_{i}")
        axis[axis_id, 1].set_title(f"{obj_name}_{i}")
        axis[axis_id, 0].legend("Translation")
        axis[axis_id, 1].legend("Rotation")
        axis[axis_id, 0].set_xlim(-1, len(estimate_progress))
        axis[axis_id, 1].set_xlim(-1, len(estimate_progress))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.7)
    plt.show()

def example_with_vizualization():
    base_path = Path(__file__).parent.parent / "datasets_aruco"
    dataset_path = base_path / "multiview"
    sam = SAM()
    frames_gt = load_data(dataset_path / "frames_gt.p")
    refined = []
    detection = Detection()
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "frames"))):
        img_path = dataset_path / "frames" / img_name
        frame = cv2.imread(str(img_path))
        frames_prediction = detection.get_marker_locations(frame, display=False)
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])
        for key in frames_prediction:
            sam.insert_T_co_detection(frames_prediction[key], key)
        sam.update_estimate()
        sam.draw_3d_estimate()
        poses = sam.get_all_T_co()
        refined.append(poses)
    objects_to_plot = ["2", "5", "7", "4", "3"]
    plot_split_results(objects_to_plot, frames_gt, [refined])

def example_on_frames_prediction():
    base_path = Path(__file__).parent.parent / "datasets"
    # dataset_path = base_path / "crackers_new"
    # dataset_path = base_path / "static_medium"
    dataset_path = base_path / "static1"
    # dataset_path = base_path / "crackers_duplicates"
    # dataset_path = base_path / "dynamic1"
    sam = SAM()
    frames_gt = load_data(dataset_path / "frames_gt.p")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    refined = []
    estimate_progress = []
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "frames"))):
        img_path = dataset_path / "frames" / img_name
        # frame = cv2.imread(str(img_path))
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])
        for key in frames_prediction[i]:
            sam.insert_T_co_detections(frames_prediction[i][key], key)
        sam.update_estimate()
        # estimate_progress.append(sam.export_current_state())
        fig = sam.draw_3d_estimate_mm()
        # fig.savefig(dataset_path/"gtsam_viz"/f'{i:04}.png')
        poses = sam.get_all_T_co()
        refined.append(poses)
    # with open(dataset_path / 'estimate_progress.p', 'wb') as file:
    #     pickle.dump(estimate_progress, file)

    with open(dataset_path / 'frames_refined_prediction.p', 'wb') as file:
        pickle.dump(refined, file)
    objects_to_plot = ["02_cracker_box", "02_cracker_box", "02_cracker_box"]#, "03_sugar_box", "07_pudding_box", "12_bleach_cleanser"]

    # plot_split_results(objects_to_plot, frames_gt, [refined])

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

def refine_ycbv_inference(DATASETS_PATH, DATASET_NAME):
    dataset_path = DATASETS_PATH/"test"/ f"{DATASET_NAME:06}"

    sam = SAM()
    frames_gt = load_scene_camera(dataset_path / "scene_camera.json")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    px_counts = load_data(dataset_path / "frames_px_counts.p")
    refined = []
    estimate_progress = []
    images = sorted(os.listdir(dataset_path / "rgb"))
    repetitions = 1
    time_each_frame = np.zeros((len(images)*repetitions))
    landmarks_each_frame = np.zeros((len(images)*repetitions))  # the ammount of landmarks recorded in the graph

    for a in range(repetitions):
        for i, img_name in enumerate(images):

            idx = a*len(images) + i
            # img_path = dataset_path / "rgb" / img_name
            start_time = time.time()
            sam.insert_odometry_measurements()
            sam.insert_T_bc_detection(np.linalg.inv(frames_gt[i]['T_cw']))
            for key in frames_prediction[i]:
                sam.insert_T_co_detections(frames_prediction[i][key], key, px_counts[i][key])
            sam.update_fls()
            # sam.update_current_estimate()

            time_each_frame[idx] = time.time() - start_time
            landmarks_each_frame[idx] = sam.all_factors_count
            # estimate_progress.append(sam.export_current_state())
            # fig = sam.draw_3d_estimate_mm()
            # fig.savefig(dataset_path/"gtsam_viz"/f'{i:04}.png')
            poses = sam.get_all_T_co()
            refined.append(poses)

    # with open(dataset_path / 'estimate_progress.p', 'wb') as file:
    #     pickle.dump(estimate_progress, file)
            print(f"\r({(idx + 1)}/{len(images)*repetitions})", end='')
    # plot_estimate_progress(estimate_progress)
    # plot_time_delays(time_each_frame, landmarks_each_frame)
    print("")
    with open(dataset_path / 'frames_refined_prediction.p', 'wb') as file:
        pickle.dump(refined, file)
    # plot_split_results(objects_to_plot, frames_gt, [refined])

def annotate_dataset(DATASETS_PATH, datasets):
    results = {}
    for DATASET_NAME in datasets:
        print(f"dataset: {DATASET_NAME}")
        dataset_path = DATASETS_PATH / "test" / f"{DATASET_NAME:06}"
        result = refine_ycbv_inference(DATASETS_PATH, DATASET_NAME)
        results[DATASET_NAME] = result
    # export_bop(convert_frames_to_bop(results), DATASETS_PATH / 'frames_refined_predictions.csv')

def merge_inferences(DATASETS_PATH, datasets, merge_from="frames_prediction.p", merge_to = 'frames_predictions.csv', dataset_name="ycbv"):
    # datasets = [48]
    results = {}
    for DATASET_NAME in datasets:
        dataset_path = DATASETS_PATH / "test" / f"{DATASET_NAME:06}"
        result = load_data(dataset_path/merge_from)
        results[DATASET_NAME] = result
    export_bop(convert_frames_to_bop(results, dataset_name), DATASETS_PATH / merge_to)

if __name__ == "__main__":
    start_time = time.time()
    dataset_name = "hope"
    # bare_minimum_example()
    # example_with_vizualization()
    # example_on_frames_prediction()
    # refine_ysbv_inference(Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv"), 50)
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo")
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "SynthStatic"
    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    # datasets = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets = [0]
    annotate_dataset(DATASET_PATH, datasets)
    merge_inferences(DATASET_PATH, datasets, "frames_refined_prediction.p", f'gtsam_{DATASET_NAME}-test_fifo_7_true_px.csv', dataset_name)
    # merge_inferences(DATASET_PATH, datasets, "frames_refined_prediction.p", 'gtsam_hopeVideo-test_fifo.csv', dataset_name)
    # merge_inferences(DATASET_PATH, datasets, "frames_prediction.p", f'cosypose_{DATASET_NAME}-test_7.csv', dataset_name)
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()