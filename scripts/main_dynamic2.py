from pathlib import Path
from Detection import Detection
import os
import cv2
# from SAM_incremental_fixed_lag_smoother import SAM
# from SAM_isam2 import SAM
# from SAM_dynamic import SAM
from SAM_dynamic2 import SAM, SAMSettings
import pickle
from compare_gt_predictions2 import plot_split_results
import time
import json
import numpy as np
from bop_tools import convert_frames_to_bop, export_bop
import matplotlib.pyplot as plt
import pinocchio as pin
import gtsam
import multiprocessing
import shutil

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

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

def refine_ycbv_inference(DATASETS_PATH, DATASET_NAME, sam_settings):
    dataset_path = DATASETS_PATH/"test"/ f"{DATASET_NAME:06}"

    sam = SAM(sam_settings)
    sam.current_time_stamp = -0.1
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
            start_time = time.time()
            if i % sam_settings.mod == 0:
            # if i % 6 == 0:
                sam.insert_odometry_measurements()
                sam.insert_T_bc_detection(np.linalg.inv(frames_gt[i]['T_cw']), timestamp=i/30)
                for key in frames_prediction[i]:
                    sam.insert_T_co_detections(frames_prediction[i][key], key, px_counts[i][key])
                sam.update_fls()
                poses = sam.get_all_T_co(timestamp=i/30)
            else:
                poses = sam.get_all_T_co(current_T_bc=np.linalg.inv(frames_gt[i]['T_cw']), timestamp=i/30)
                pass

            time_each_frame[idx] = time.time() - start_time
            landmarks_each_frame[idx] = sam.all_factors_count
            refined.append(poses)
            print(f"\r({(idx + 1)}/{len(images)*repetitions})", end='')

    print("")
    with open(dataset_path / 'frames_refined_prediction.p', 'wb') as file:
        pickle.dump(refined, file)
    return refined


def annotate_dataset(DATASETS_PATH, datasets, cov1=0.001, cov2=0.0001, ort=20, tvt=0.00001, Rvt=0.0002):
    results = {}
    for DATASET_NAME in datasets:
        print(f"dataset: {DATASET_NAME}")
        dataset_path = DATASETS_PATH / "test" / f"{DATASET_NAME:06}"
        result = refine_ycbv_inference(DATASETS_PATH, DATASET_NAME, ort=ort, tvt=tvt, Rvt=Rvt, cov1=cov1, cov2=cov2)
        results[DATASET_NAME] = result
    # export_bop(convert_frames_to_bop(results), DATASETS_PATH / 'frames_refined_predictions.csv')

def anotate_dataset_parallel_safe(dataset_name, DATASETS_PATH, datasets, sam_settings, output_name):
    results = {}
    print(f"datasets: {datasets}")
    for DATASET_NAME in datasets:
        result = refine_ycbv_inference(DATASETS_PATH, DATASET_NAME, sam_settings)
        results[DATASET_NAME] = result
    export_bop(convert_frames_to_bop(results, dataset_name), DATASETS_PATH / "ablation" / output_name)
    print(f"saving results to: {DATASETS_PATH / 'ablation' / output_name}")


def annotate_dataset_multithreaded(DATASETS_PATH, datasets, cov1, cov2, ort, tvt, Rvt):
    procs = []
    start_time = time.time()
    print(f"anotating datasets: {datasets}")
    for DATASET_NAME in datasets:
        proc = multiprocessing.Process(target=refine_ycbv_inference, args=(DATASETS_PATH, DATASET_NAME,cov1, cov2, ort, tvt, Rvt))
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
    # refine_ysbv_inference(Path("/mnt/Data/HappyPose_Data/bop_datasets/ycbv"), 50)
    # DATASETS_PATH = Path("/mnt/Data/HappyPose_Data/bop_datasets/ycbv")
    # DATASETS_PATH = Path("/mnt/Data/HappyPose_Data/bop_datasets/hopeVideo")
    DATASETS_PATH = Path("/mnt/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "SynthStatic"
    DATASET_NAME = "hopeVideo"
    # DATASET_NAME = "SynthDynamic"
    # DATASET_NAME = "SynthDynamicOcclusion"
    # DATASET_NAME = "SynthTest"

    DATASET_PATH = DATASETS_PATH / DATASET_NAME
    # datasets = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets = [0, 1, 2]
    # datasets = [0]
    __refresh_dir(DATASETS_PATH / DATASET_NAME / "ablation")
    pool = multiprocessing.Pool(processes=15)

    # for ws in [2, 5, 10, 20]:
    for mod in [1]:
        for ws in [3]:
            for ort in [10]:
                # for tvt in [0.0000025, 0.000005, 0.00001]:
                for tvt in [0.00025]:
                    for Rvt in [0.0022]:
                    # for Rvt in [0.00125, 0.0015, 0.00175]:
                    #     for cov_drift_lin_vel in [1, 0.5, 0.1, 0.05, 0.01, 0.001]:
                        for cov_drift_lin_vel in [0.1]:
                            # for cov_drift_ang_vel in [2.0, 1.0, 0.5]:
                            for cov_drift_ang_vel in [1]:
                                for cov2_t in [0.000000001]:
                                    for cov2_R in [0.000000001]:
                                    # for hyster in [25, 200, 800]:
                                        for hyster in [1]:
                                            sam_settings = SAMSettings(mod=mod,
                                                                       window_size=ws,
                                                                       cov_drift_lin_vel=cov_drift_lin_vel,
                                                                       cov_drift_ang_vel=cov_drift_ang_vel,
                                                                       cov2_t=cov2_t,
                                                                       cov2_R=cov2_R,
                                                                       outlier_rejection_treshold=ort,
                                                                       t_validity_treshold=tvt,
                                                                       R_validity_treshold=Rvt,
                                                                       hysteresis_coef=hyster,
                                                                       velocity_prior_sigma=10,
                                                                       reject_overlaps=0.05)
                                            print(f"{mod},{ort}, {tvt:.8f}, {Rvt:.8f}, {cov_drift_lin_vel:.8f}, {cov_drift_ang_vel:.8f}, {cov2_t:.8f}, {cov2_R:.8f}")
                                            output_name = f'gtsam_{DATASET_NAME}-test_{mod}_{str(sam_settings)}_.csv'

                                            # pool.apply_async(anotate_dataset_parallel_safe, args=(dataset_type, DATASETS_PATH/DATASET_NAME, datasets, sam_settings, output_name))
                                            anotate_dataset_parallel_safe(dataset_type, DATASETS_PATH/DATASET_NAME, datasets, sam_settings, output_name)
    pool.close()
    pool.join()

    # merge_inferences(DATASET_PATH, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "frames_prediction_mod6.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)
    merge_inferences(DATASET_PATH, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "frames_prediction.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()