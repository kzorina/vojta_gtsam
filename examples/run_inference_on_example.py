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

def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    if torch.cuda.is_available():
        observation.cuda()
    return observation

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
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


def rendering(predictions, example_dir):
    object_dataset = make_object_dataset(example_dir)

    labels = predictions.infos["label"]
    idxs = predictions.infos["coarse_instance_idx"]
    # rendering
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
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


def save_predictions(example_dir, renderings):
    rgb_render = renderings.rgb
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    # render_prediction_wrt_camera calls BulletSceneRenderer.render_scene using only one camera at pose Identity and return only rgb values
    # BulletSceneRenderer.render_scene: gets a "object list" (prediction like object), a list of camera infos (with Km pose, res) and renders
    # a "camera observation" for each camera/viewpoint
    # Actually, renders: rgb, mask, depth, near, far
    #rgb_render = render_prediction_wrt_camera(renderer, preds, cam)
    mask = ~(rgb_render.sum(axis=-1) == 0)
    alpha = 0.1
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = rgb_render[mask]

    # make the image background a bit fairer than the render
    rgb_overlay = np.zeros_like(rgb_render)
    rgb_overlay[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
    rgb_overlay[mask] = rgb_render[mask] * 0.9 + 255 * 0.1
    plotter = BokehPlotter()
    comparisson_img = cv2.cvtColor(np.concatenate((rgb, rgb_overlay), axis=1), cv2.COLOR_BGR2RGB)

    cv2.imwrite('rgb_overlay.png', rgb_overlay)
    cv2.imwrite('rgb.png', rgb)
    cv2.imwrite('comparisson_img.png', comparisson_img)
    time.sleep(1)
    cv2.imshow('comparisson_img', comparisson_img)
    cv2.waitKey(0)


def run_inference(
    example_dir: Path,
    model_name: str,
    dataset_to_use: str,
) -> None:
    observation = load_observation_tensor(example_dir)
    # TODO: remove this wrapper from code base
    CosyPose = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)
    predictions = CosyPose.inference(observation)
    renderings = rendering(predictions, example_dir)
    save_predictions(example_dir, renderings)


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    # parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--dataset", type=str, default="ycbv")
    #parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true", default=True)
    #parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    # data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
    # assert data_dir
    example_dir = Path(__file__).parent.parent / "datasets" / args.example_name
    dataset_to_use = args.dataset  # tless or ycbv


    if args.run_inference:
        run_inference(example_dir, None, dataset_to_use)
