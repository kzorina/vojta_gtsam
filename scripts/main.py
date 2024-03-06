from pathlib import Path
from Detection import Detection
import os
import cv2
# from SAM_incremental_fixed_lag_smoother import SAM
# from SAM_isam2 import SAM
# from SAM_dynamic import SAM
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
from collections import defaultdict
import multiprocessing

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

def load_scene_camera(path):
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

def refine_ycbv_inference(DATASETS_PATH, DATASET_NAME, elt=0.0001, ort=20, tvt=0.00001, Rvt=0.0002):
    dataset_path = DATASETS_PATH/"test"/ f"{DATASET_NAME:06}"

    sam = SAM()
    sam.outlier_rejection_treshold = ort
    sam.t_validity_treshold = tvt
    sam.R_validity_treshold = Rvt
    sam.elapsed_time = elt

    frames_camera = load_scene_camera(dataset_path / "scene_camera.json")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    frames_gt = load_scene_gt(dataset_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
    # frames_prediction = frames_gt  ### TODO: for test purpose only, remove afterwards!!!!
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
            if i % 1 == 0:
            # if i % 1 == 0:
                sam.insert_odometry_measurements()
                sam.insert_T_bc_detection(np.linalg.inv(frames_camera[i]['T_cw']))
                for key in frames_prediction[i]:
                    sam.insert_T_co_detections(frames_prediction[i][key], key, px_counts[i][key])
                sam.update_fls()
                poses = sam.get_all_T_co()
            else:
                poses = sam.get_all_T_co(current_T_bc=np.linalg.inv(frames_camera[i]['T_cw']))
            # sam.update_current_estimate()

            time_each_frame[idx] = time.time() - start_time
            landmarks_each_frame[idx] = sam.all_factors_count
            # estimate_progress.append(sam.export_current_state())
            # fig = sam.draw_3d_estimate_mm()
            # fig.savefig(dataset_path/"gtsam_viz"/f'{i:04}.png')
            # poses = sam.get_all_T_co()
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

def annotate_dataset(DATASETS_PATH, datasets,elt, ort, tvt, Rvt):
    results = {}
    start_time = time.time()
    for DATASET_NAME in datasets:
        print(f"dataset: {DATASET_NAME}")
        dataset_path = DATASETS_PATH / "test" / f"{DATASET_NAME:06}"
        result = refine_ycbv_inference(DATASETS_PATH, DATASET_NAME,elt, ort=ort, tvt=tvt, Rvt=Rvt)
        results[DATASET_NAME] = result
    print(f"elapsed_time:{time.time() - start_time}")
def annotate_dataset_multithreaded(DATASETS_PATH, datasets,elt, ort, tvt, Rvt):
    procs = []
    start_time = time.time()
    print(f"anotating datasets: {datasets}")
    for DATASET_NAME in datasets:
        proc = multiprocessing.Process(target=refine_ycbv_inference, args=(DATASETS_PATH, DATASET_NAME,elt, ort, tvt, Rvt))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    print(f"elapsed_time:{time.time() - start_time}")
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
    dataset_type = "hope"
    # bare_minimum_example()
    # example_with_vizualization()
    # example_on_frames_prediction()
    # refine_ysbv_inference(Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv"), 50)
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo")
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "SynthStatic"
    DATASET_NAME = "hopeVideo"
    # DATASET_NAME = "SynthDynamic"

    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    # datasets = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets = [0]
    for ort in [1, 5, 10]: #  [5]:
        for tvt in [1e-05, 5e-06, 1.e-06]: #  [(0.00000385, 0.0002)]:
            for Rvt in [1e-03, 5e-04, 1e-04]: #  [(0.00000385, 0.0002)]:
                for elt in [0.0001]:
                    #  last iters:
                    #  10, 0.0000100000, 0.0010000000
                    #  10, 0.0000100000, 0.0005000000
                    #  10, 0.0000100000, 0.0001000000 unfinished
                    print(f"{ort}, {tvt:.10f}, {Rvt:.10f}")
                    # annotate_dataset(DATASET_PATH, datasets,elt=elt, ort=ort, tvt=tvt, Rvt=Rvt)
                    annotate_dataset_multithreaded(DATASET_PATH, datasets,elt=elt, ort=ort, tvt=tvt, Rvt=Rvt)
                    merge_inferences(DATASET_PATH, datasets, "frames_refined_prediction.p", f'gtsam_{DATASET_NAME}-test_mod1_{elt}_{ort}_{tvt:.10f}_{Rvt:.10f}_.csv', dataset_type)
    # annotate_dataset(DATASET_PATH, datasets)
    # merge_inferences(DATASET_PATH, datasets, "frames_refined_prediction.p", f'gtsam_{DATASET_NAME}-test.csv', dataset_type)
    # merge_inferences(DATASET_PATH, datasets, "frames_refined_prediction.p", 'gtsam_hopeVideo-test_fifo.csv', dataset_type)
    merge_inferences(DATASET_PATH, datasets, "frames_prediction.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()