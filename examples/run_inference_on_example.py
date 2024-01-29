# Standard Library
import argparse
import json
import os

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


import cv2

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


logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_observation(dataset_dir: Path, img_path) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    if os.path.exists(dataset_dir / "camera_data.json"):
        camera_data = CameraData.from_json((dataset_dir / "camera_data.json").read_text())
    else:
        camera_data = CameraData.from_json((dataset_dir.parent / "camera_data.json").read_text())
    rgb = np.array(Image.open(img_path), dtype=np.uint8)
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


def rendering(predictions, dataset_dir):
    object_dataset = make_object_dataset(dataset_dir)

    labels = predictions.infos["label"]
    idxs = predictions.infos["coarse_instance_idx"]
    # rendering
    if os.path.exists(dataset_dir / "camera_data.json"):
        camera_data = CameraData.from_json((dataset_dir / "camera_data.json").read_text())
    else:
        camera_data = CameraData.from_json((dataset_dir.parent / "camera_data.json").read_text())
    object_labels = [object_dataset.list_objects[i].label for i in range(len(object_dataset.list_objects))]
    camera_data.TWC = Transform(np.eye(4))
    renderer = Panda3dSceneRenderer(object_dataset)
    # Data necessary for image rendering
    object_datas = []
    for idx, rough_label in zip(idxs, labels):
        pred = predictions.poses[idx].numpy()
        label = YCBV_OBJECT_NAMES[rough_label.split("-")[1]]
        object_datas.append(ObjectData(label=label, TWO=Transform(pred)))
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


def save_prediction_img(output_path, img_name, rgb, rgb_render):
    mask = ~(rgb_render.sum(axis=-1) == 0)
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = rgb_render[mask]

    rgb_overlay = np.zeros_like(rgb_render)
    rgb_overlay[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
    rgb_overlay[mask] = rgb_render[mask] * 0.9 + 255 * 0.1
    comparison_img = cv2.cvtColor(np.concatenate((rgb, rgb_overlay), axis=1), cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path/img_name), comparison_img)

def save_preditions_data(output_path, all_predictions):
    frames = []
    for predictions in all_predictions:
        entry = {}
        poses_tensor:torch.Tensor = predictions.tensors["poses"]
        for idx, label in enumerate(predictions.infos["label"]):
            pose = poses_tensor[idx].numpy()
            obj_name = YCBV_OBJECT_NAMES[label.split("-")[1]]
            if obj_name not in entry:
                entry[obj_name] = []
            entry[obj_name].append(pose)
        frames.append(entry)
    with open(output_path, 'wb') as file:
        pickle.dump(frames, file)
    print(f"data saved to {output_path}")
    return frames


def run_inference(dataset_dir: Path, dataset_to_use: str) -> None:

    img_names = sorted(os.listdir(dataset_dir / "frames"))
    CosyPose = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)
    start_time = time.time()

    all_predictions = []
    for img_name in img_names:
        frame_processing_time_start = time.time()
        img_path = dataset_dir / "frames" / img_name
        rgb, depth, camera_data = load_observation(dataset_dir, img_path)
        observation = data_to_observation(rgb, depth, camera_data)
        predictions = CosyPose.inference(observation)
        renderings = rendering(predictions, dataset_dir)
        save_prediction_img(dataset_dir / "output", img_name, rgb, renderings.rgb)
        all_predictions.append(predictions)
        print(f"inference successfully. {(time.time() - frame_processing_time_start):9.4f}s")
    save_preditions_data(dataset_dir/"frames_prediction.p", all_predictions)
    print(f"runtime: {(time.time() - start_time):.2f}s for {len(img_names)} images")

def main():
    set_logging_level("info")
    # dataset_name = "crackers_new"
    dataset_name = "test1"
    # dataset_name = "dynamic1"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name

    run_inference(dataset_path, "ycbv")

if __name__ == "__main__":
    main()
