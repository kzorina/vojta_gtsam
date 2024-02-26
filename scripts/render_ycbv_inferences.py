# Standard Library
import argparse
import json
import os
import gc
########################
# Add cosypose to my path -> dirty
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import cosypose
import cv2

# Third Party
import numpy as np
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import pickle
# from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)

#from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from happypose.pose_estimators.cosypose.cosypose.visualization.singleview import (
    render_prediction_wrt_camera,
)

# MegaPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.lib3d.transform import Transform

# HappyPose
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay

########################
import shutil
from bop_tools import convert_frames_to_bop, export_bop

import cv2
import pinocchio as pin

YCBV_OBJECT_NAMES = {"obj_000001": "01_master_chef_can",
    "obj_000002": "02_cracker_box",
    "obj_000003": "03_sugar_box",
    "obj_000004": "04_tomatoe_soup_can",
    "obj_000005": "05_mustard_bottle",
    "obj_000006": "06_tuna_fish_can",
    "obj_000007": "07_pudding_box",
    "obj_000008": "08_gelatin_box",
    "obj_000009": "09_potted_meat_can",
    "obj_000010": "10_banana",
    "obj_000011": "11_pitcher_base",
    "obj_000012": "12_bleach_cleanser",
    "obj_000013": "13_bowl",
    "obj_000014": "14_mug",
    "obj_000015": "15_power_drill",
    "obj_000016": "16_wood_block",
    "obj_000017": "17_scissors",
    "obj_000018": "18_large_marker",
    "obj_000019": "19_large_clamp",
    "obj_000020": "20_extra_large_clamp",
    "obj_000021": "21_foam_brick"}

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


# logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_observation(dataset_dir: Path, img_path, K = None) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    if K is None:
        if os.path.exists(dataset_dir / "camera_data.json"):
            camera_data = CameraData.from_json((dataset_dir / "camera_data.json").read_text())
        else:
            camera_data = CameraData.from_json((dataset_dir.parent / "camera_data.json").read_text())
    else:
        camera_data = CameraData
        camera_data.K = K

    rgb = np.array(Image.open(img_path), dtype=np.uint8)
    camera_data.resolution = rgb.shape[:2]
    assert rgb.shape[:2] == camera_data.resolution
    depth = None
    return rgb, depth, camera_data

def data_to_observation(rgb, depth, camera_data) -> ObservationTensor:
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    if torch.cuda.is_available():
        observation.cuda()
    return observation

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    if os.path.exists(example_dir / "meshes"):
        object_dirs = (example_dir / "meshes").iterdir()
    else:
        object_dirs = (example_dir.parent / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def rendering(predictions, renderer, K, resolution=(640, 480)):
    camera_data = CameraData
    camera_data.K = K
    camera_data.TWC = Transform(np.eye(4))
    camera_data.resolution = resolution
    object_datas = []
    for label in predictions:
        if label != "Camera":
            for T_co in predictions[label]:
                object_datas.append(ObjectData(label=label, TWO=Transform(T_co)))
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((0.6, 0.6, 0.6, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]
    return renderings


def save_prediction_img(output_path, img_name, rgb, rgb_renders):
    overlays = []
    for rgb_render in rgb_renders:
        mask = ~(rgb_render.sum(axis=-1) == 0)
        rgb_n_render = rgb.copy()
        rgb_n_render[mask] = rgb_render[mask]

        rgb_overlay = np.zeros_like(rgb_render)
        rgb_overlay[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
        rgb_overlay[mask] = rgb_render[mask] * 0.9 + 255 * 0.1
        overlays.append(rgb_overlay)
    comparison_img = cv2.cvtColor(np.concatenate([rgb]+ overlays, axis=1), cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path/img_name), comparison_img)

def predictions_to_dict(predictions):
    entry = {}
    poses_tensor: torch.Tensor = predictions.tensors["poses"]
    for idx, label in enumerate(predictions.infos["label"]):
        pose = poses_tensor[idx].numpy()
        obj_name = YCBV_OBJECT_NAMES[label.split("-")[1]]
        if obj_name not in entry:
            entry[obj_name] = []
        entry[obj_name].append(pose)
    return entry

def save_preditions_data(output_path, all_predictions):
    with open(output_path, 'wb') as file:
        pickle.dump(all_predictions, file)
    print(f"data saved to {output_path}")

def get_tensors():
    objects = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.get_device() == 0:
                    objects.append(obj)
        except:
            pass
    return objects

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

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

def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_scene_gt(path, label_list = None):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        frame = i+1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            if label_list is not None:
                entry[label_list[obj_id-1]] = [T_cm]
            else:
                entry[obj_id] = [T_cm]
        parsed_data.append(entry)
    return parsed_data

def main():
    set_logging_level("info")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "hopeVideo"
    DATASET_NAME = "SynthStatic"
    # DATASET_NAME = "SynthDynamic"
    DATASET_PATH = DATASETS_PATH/DATASET_NAME
    MESHES_PATH = DATASETS_PATH/DATASET_NAME/"meshes"
    SCENES_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]

    object_dataset = make_object_dataset(MESHES_PATH)
    renderer = Panda3dSceneRenderer(object_dataset)

    for scene_name in SCENES_NAMES[0:1]:
        print(f"\n{scene_name}:")
        dataset_path = DATASET_PATH / "test" / scene_name
        output_dir = dataset_path / "output_fifo"
        __refresh_dir(output_dir)
        scene_camera = load_scene_camera(dataset_path / "scene_camera.json")
        # scene_gt = load_scene_gt(dataset_path / "scene_gt.json", list(YCBV_OBJECT_NAMES.values()))
        scene_gt = load_scene_gt(dataset_path / "scene_gt.json", list(HOPE_OBJECT_NAMES.values()))
        img_names = sorted(os.listdir(dataset_path / "rgb"))
        frames_prediction = load_data(dataset_path / "frames_prediction.p")
        frames_refined_prediction = load_data(dataset_path / "frames_refined_prediction.p")

        # T_co_0 = np.array(((0, 0, 1, 0),
        #                      (0, 1, 0, 0),
        #                      (1, 0, 0, 0.8),
        #                      (0, 0, 0, 1)))
        T_wc_0 = np.linalg.inv(scene_camera[0]["T_cw"])
        # T_co_0 = scene_gt[0]
        # T_wo = T_wc_0 @ T_co_0
        for i in range(0, len(img_names), 1):
            img_name = img_names[i]
            rgb = np.array(Image.open(dataset_path/"rgb"/img_name), dtype=np.uint8)
            K = scene_camera[i]["cam_K"]
            # T_cw = scene_camera[i]["T_cw"]
            # T_co = T_cw@T_wo
            # artificial_prediction = {"03_sugar_box": [T_co]}
            # renderings = rendering(artificial_prediction, renderer, K, rgb.shape[:2])
            # renderings = rendering(scene_gt[i], renderer, K, rgb.shape[:2])
            # renderings = rendering(frames_prediction[i], renderer, K, rgb.shape[:2])
            renderings_gtsam = rendering(frames_refined_prediction[i], renderer, K, rgb.shape[:2])
            renderings_cosypose = rendering(frames_prediction[i], renderer, K, rgb.shape[:2])
            # save_prediction_img(output_dir, img_name, rgb, [renderings_cosypose.rgb])
            save_prediction_img(output_dir, img_name, rgb, [renderings_cosypose.rgb, renderings_gtsam.rgb])
            print(f"\r({i+1}/{len(img_names)})", end='')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nelapsed time: {time.time() - start_time:.2f} s")
