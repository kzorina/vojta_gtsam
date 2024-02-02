from pathlib import Path
from Detection import Detection
import os
import cv2
from SAM import SAM
import pickle
from compare_gt_predictions2 import plot_split_results
import time
import json
import numpy as np
from bop_tools import convert_frames_to_bop, export_bop

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


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
    for i in sorted(data.keys()):
        entry = {}
        entry["cam_K"] = np.array(data[i]["cam_K"]).reshape((3, 3))
        T_wc = np.zeros((4, 4))
        T_wc[:3, :3] = np.array(data[i]["cam_R_w2c"]).reshape((3, 3))
        T_wc[3, :3] = np.array(data[i]["cam_t_w2c"])
        entry["Camera"] = T_wc
        parsed_data.append(entry)
    return parsed_data

def refine_ysbv_inference():
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    DATASET_NAME = 48
    dataset_path = DATASETS_PATH/"test"/ f"{DATASET_NAME:06}"

    sam = SAM()
    frames_gt = load_scene_camera(dataset_path / "scene_camera.json")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    refined = []
    estimate_progress = []
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "rgb"))):
        img_path = dataset_path / "rgb" / img_name
        # frame = cv2.imread(str(img_path))
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])
        for key in frames_prediction[i]:
            sam.insert_T_co_detections(frames_prediction[i][key], key)
        sam.update_estimate()
        # estimate_progress.append(sam.export_current_state())
        # fig = sam.draw_3d_estimate_mm()
        # fig.savefig(dataset_path/"gtsam_viz"/f'{i:04}.png')
        poses = sam.get_all_T_co()
        refined.append(poses)
    # with open(dataset_path / 'estimate_progress.p', 'wb') as file:
    #     pickle.dump(estimate_progress, file)
    with open(dataset_path / 'frames_refined_prediction.p', 'wb') as file:
        pickle.dump(refined, file)
    export_bop(convert_frames_to_bop({DATASET_NAME:refined}, dataset_path / 'frames_refined_prediction.csv'))
    # objects_to_plot = ["02_cracker_box", "02_cracker_box", "02_cracker_box"]#, "03_sugar_box", "07_pudding_box", "12_bleach_cleanser"]

    # plot_split_results(objects_to_plot, frames_gt, [refined])

if __name__ == "__main__":
    start_time = time.time()
    # bare_minimum_example()
    # example_with_vizualization()
    # example_on_frames_prediction()
    refine_ysbv_inference()
    print(f"elapsed time: {time.time() - start_time:.2f} s")
    # main()