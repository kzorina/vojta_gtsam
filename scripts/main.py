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

if __name__ == "__main__":
    # bare_minimum_example()
    example_with_vizualization()
    # main()