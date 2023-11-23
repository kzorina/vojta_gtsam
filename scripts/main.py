from pathlib import Path
from Detection import Detection
import os
import cv2
from SAM import SAM
import pickle
from compare_gt_predictions2 import plot_split_results
from Layers import Layers
from Vizualize import draw_layers
def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def example_with_vizualization():
    base_path = Path(__file__).parent.parent / "datasets"
    dataset_path = base_path / "multiview"
    sam = SAM()
    frames_gt = load_data(dataset_path / "frames_gt.p")
    nonrefined = []
    refined = []
    detection = Detection()
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "frames"))):
        img_path = dataset_path / "frames" / img_name
        frame = cv2.imread(str(img_path))

        frames_prediction = detection.get_marker_locations(frame, display=False)
        nonrefined.append(frames_prediction)
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])
        for key in frames_prediction:
            sam.insert_T_co_detection(frames_prediction[key], key)
        sam.update_estimate()
        sam.draw_3d_estimate()
        poses = sam.get_all_T_co()
        refined.append(poses)
    objects_to_plot = ["2", "5", "7", "4", "3"]
    plot_split_results(objects_to_plot, frames_gt, [refined])
    layers = Layers()
    all_T_bo = sam.get_all_T_bo()
    for key in all_T_bo:
        layers.insert_cube(all_T_bo[key], key)
    draw_layers(layers)

def bare_minimum_example():
    base_path = Path(__file__).parent.parent / "datasets"  # not interesting
    dataset_path = base_path / "multiview"  # not interesting
    sam = SAM()
    frames_gt = load_data(dataset_path / "frames_gt.p")  # not interesting
    detection = Detection()
    for i, img_name in enumerate(sorted(os.listdir(dataset_path / "frames"))):
        img_path = dataset_path / "frames" / img_name  # not interesting
        frame = cv2.imread(str(img_path))  # not interesting
        frames_prediction = detection.get_marker_locations(frame, display=False)  # initial object poses predictions T_co
        sam.insert_T_bc_detection(frames_gt[i]['Camera'])  # camera to robot base estimate added to SAM
        for key in frames_prediction:
            sam.insert_T_co_detection(frames_prediction[key], key)  # initial object poses predictions added to SAM
        sam.update_estimate()  # refine SAM estimate
    layers = Layers()
    all_T_bo = sam.get_all_T_bo()
    for key in all_T_bo:
        layers.insert_cube(all_T_bo[key], key)  # add refined estimates to layers
    cubes = layers.get_grabbable_cubes()  # get a sorted list of cubes that can be grabbed. first cube is on top - grab that one first.
    ungrabbable_cubes = layers.resolve_grabbability(0)
    first_cube = cubes[0]
    grip = first_cube.get_valid_grips()[0]
    T_bg = layers.T_bt @ first_cube.T_to @ grip.T_og  # grip to robot base transformation. If it is too high/low, adjust z_offset in Grip.py.
    print(f"grip_T_bg:{T_bg}")
    print(ungrabbable_cubes)
if __name__ == "__main__":
    # bare_minimum_example()
    example_with_vizualization()
    # main()