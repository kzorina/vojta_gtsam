import cv2
import numpy as np
from pathlib import Path
import pinocchio as pin
import pickle
from typing import Dict
import os
from Vizualize import draw_layer
from compare_gt_predictions2 import plot_split_results

class Detection():
    def __init__(self):
        self.cube_size = 0.05  # in meters
        self.camera_matrix = np.array([[614.193, 0, 326.268],
                             [0, 614.193, 238.851],
                              [0, 0, 1]])
        self.camera_distortion = np.array([0.0, 0.0, 0.0, 0.0])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def get_marker_locations(self, frame: np.ndarray, display: bool = False) -> Dict[str, np.ndarray]:
        """
        :param frame: rgb image
        :param display: if true, the detected markers will be displayed overlaying the frame .
        :return: Dictionary of detected markers. {"1": T_bo, "2": T_bo...}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = self.detector.detectMarkers(gray)

        if display:
            new_frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
            new_frame = cv2.aruco.drawDetectedMarkers(image=new_frame, corners=rejected_img_points, borderColor=(0, 0, 255))
        ret = {}
        if len(corners) > 0:
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, self.camera_matrix, self.camera_distortion)
                T_co = pin.SE3(pin.exp3(rvec.reshape(3)), tvec.reshape(3))
                ret[f"{ids[i][0]}"] = T_co.homogeneous
                if display:
                    cv2.drawFrameAxes(new_frame, self.camera_matrix, self.camera_distortion, rvec, tvec, 0.05)
        if display:
            cv2.imshow("press q for next frame", new_frame)
            cv2.waitKey(0)
        return ret

def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    base_path = Path(__file__).parent.parent / "datasets"
    dataset_path = base_path / "multiview"

    for img_name in sorted(os.listdir(dataset_path / "frames")):
        # img_path = str(dataset_path / "frames" / "000001.png")
        img_path = dataset_path / "frames" / img_name
        frame = cv2.imread(str(img_path))
        detection = Detection()
        frames_prediction = detection.get_marker_locations(frame, display=True)

    # frames_gt = load_data(dataset_path/"frames_gt.p")
    # objects_to_plot = ["2", "5", "7", "4", "3"]
    # plot_split_results(objects_to_plot, frames_gt, [frames_prediction])


if __name__ == "__main__":
    main()