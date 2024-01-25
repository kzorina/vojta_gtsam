from pathlib import Path
from Detection import Detection
import os
import cv2
from SAM import SAM
import pickle
from compare_gt_predictions2 import plot_split_results

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
    dataset_path = base_path / "crackers_duplicates"
    sam = SAM()
    frames_gt = load_data(dataset_path / "frames_gt.p")
    frames_prediction = load_data(dataset_path / "frames_prediction.p")
    refined = []
    estimate_progress = []
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "frames"))):
        img_path = dataset_path / "frames" / img_name
        frame = cv2.imread(str(img_path))
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])
        for key in frames_prediction[i]:
            sam.insert_T_co_detections(frames_prediction[i][key], key)
        sam.update_estimate()
        estimate_progress.append(sam.export_current_state())
        sam.draw_3d_estimate()
        # poses = sam.get_all_T_co()  # TODO: make compatible with duplicates
        # refined.append(poses)  # TODO: make compatible with duplicates
    # objects_to_plot = ["2", "5", "7", "4", "3"]
    if os.path.exists(dataset_path / 'estimate_progress.p'):
        os.remove(dataset_path / 'estimate_progress.p')
    with open(dataset_path / 'estimate_progress.p', 'ab') as file:
        pickle.dump(estimate_progress, file)
    objects_to_plot = ["02_cracker_box", "03_sugar_box", "07_pudding_box", "12_bleach_cleanser"]
    # plot_split_results(objects_to_plot, frames_gt, [refined])  # TODO: make compatible with duplicates

if __name__ == "__main__":
    # bare_minimum_example()
    # example_with_vizualization()
    example_on_frames_prediction()
    # main()