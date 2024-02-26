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

OBJECT_NAMES = HOPE_OBJECT_NAMES

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

def rendering(predictions, dataset_dir, renderer, K=None):
    labels = predictions.infos["label"]
    idxs = predictions.infos["coarse_instance_idx"]
    # rendering
    if K is None:
        if os.path.exists(dataset_dir / "camera_data.json"):
            camera_data = CameraData.from_json((dataset_dir / "camera_data.json").read_text())
        else:
            camera_data = CameraData.from_json((dataset_dir.parent / "camera_data.json").read_text())
    else:
        camera_data = CameraData
        camera_data.K = K
    camera_data.TWC = Transform(np.eye(4))
    # Data necessary for image rendering
    object_datas = []
    for idx, rough_label in zip(idxs, labels):
        pred = predictions.poses[idx].numpy()
        label = OBJECT_NAMES[rough_label.split("-")[1]]
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
        # copy_arrays=True,
        copy_arrays=False,
    )[0]
    return renderings


def save_prediction_img(output_path, img_name, rgb, rgb_render, bboxes):
    mask = ~(rgb_render.sum(axis=-1) == 0)
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = rgb_render[mask]
    rgb_overlay = np.zeros_like(rgb_render)
    rgb_overlay[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
    rgb_overlay[mask] = rgb_render[mask] * 0.9 + 255 * 0.1
    # fixed_bboxes =
    for bbox in bboxes:
        cv2.rectangle(rgb_overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # cv2.rectangle(rgb_overlay, (100, 100), (200, 200), (0, 255, 0), 2)
    comparison_img = cv2.cvtColor(np.concatenate((rgb, rgb_overlay), axis=1), cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path/img_name), comparison_img)

def predictions_to_dict(predictions, px_counts):
    entry = {}
    entry_px_count = {}
    poses_tensor: torch.Tensor = predictions.tensors["poses"]
    for idx, label in enumerate(predictions.infos["label"]):
        pose = poses_tensor[idx].numpy()
        px_count = px_counts[idx]
        obj_name = OBJECT_NAMES[label.split("-")[1]]
        if obj_name not in entry:
            entry[obj_name] = []
            entry_px_count[obj_name] = []
        entry[obj_name].append(pose)
        entry_px_count[obj_name].append(px_count)
    return entry, entry_px_count

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

def get_scene_camera(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = {}
        entry["cam_K"] = np.array(data[str(i+1)]["cam_K"]).reshape((3, 3))
        T_wc = np.zeros((4, 4))
        T_wc[:3, :3] = np.array(data[str(i+1)]["cam_R_w2c"]).reshape((3, 3))
        T_wc[3, :3] = np.array(data[str(i+1)]["cam_t_w2c"])
        T_wc[3, 3] = 1
        entry["T_wc"] = T_wc
        parsed_data.append(entry)
    return parsed_data


def run_inference(dataset_dir: Path, CosyPose, object_dataset) -> None:
    DATASET_NAME = int(dataset_dir.name)

    img_names = sorted(os.listdir(dataset_dir / "rgb"))
    # renderer = Panda3dSceneRenderer(object_dataset)
    scene_camera = get_scene_camera(dataset_dir/"scene_camera.json")

    start_time = time.time()
    all_predictions = []
    all_px_counts = []
    for i in range(len(img_names)):
        K = scene_camera[i]["cam_K"]

        img_name = img_names[i]
        frame_processing_time_start = time.time()
        img_path = dataset_dir / "rgb" / img_name
        rgb, depth, camera_data = load_observation(dataset_dir, img_path, K=K)
        observation = data_to_observation(rgb, depth, camera_data)


        # detections = CosyPose.detector.get_detections(observation, output_masks=True)
        # final_preds, all_preds = CosyPose.pose_predictor.run_inference_pipeline(observation, detections)
        # predictions = final_preds.cpu()
        predictions, extra_data = CosyPose.inference(observation, output_masks=True)
        # cosypose is a CosyPoseWrapper object

        # renderings = rendering(predictions, dataset_dir, renderer, K=K)
        # save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, renderings.rgb, predictions.tensors['boxes_crop'].data.numpy())

        # masks = extra_data['masks'].sum(0).cpu().numpy().astype(np.uint8)*20
        # masks_rgb = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)
        # save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, masks_rgb, predictions.tensors['boxes_crop'].data.numpy())
        if 'px_count' in extra_data:
            poses, bboxes = predictions_to_dict(predictions, extra_data['px_count'])
        else:
            poses, bboxes = predictions_to_dict(predictions, np.array([]))
        all_predictions.append(poses)
        all_px_counts.append(bboxes)
        del observation
        # del predictions
        # # torch.cuda.empty_cache()
        # gc.collect()
        print(f"\r({i+1}/{len(img_names)})"
              f" inferenced successfully."
              f"  {(time.time() - frame_processing_time_start):.4f}s"
              f" ({(time.time() - start_time)/60:.2f}min)",
              end='')

    print(f"\nruntime: {(time.time() - start_time):.2f}s for {len(img_names)} images")
    save_preditions_data(dataset_dir/"frames_prediction.p", all_predictions)
    save_preditions_data(dataset_dir/"frames_px_counts.p", all_px_counts)
    # export_bop(convert_frames_to_bop({DATASET_NAME: all_predictions}), dataset_dir / 'frames_prediction.csv')


def main():
    #  in case of OOM error try this:
    #  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
    # torch.cuda.empty_cache()
    set_logging_level("info")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo")
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthStatic")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthDynamic")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    MESHES_PATH = DATASETS_PATH/"meshes"

    # DATASET_NAMES = ["000048", "000049", "000050", "000051", "000052", "000053", "000054", "000055", "000056", "000057", "000058", "000059"]  # ycbv
    DATASET_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]  # hope, synth
    object_dataset = make_object_dataset(MESHES_PATH)
    # CosyPose = CosyPoseWrapper(dataset_name="ycbv", n_workers=8)
    CosyPose = CosyPoseWrapper(dataset_name="hope", n_workers=8)
    # dataset_name = "crackers_new"
    # dataset_name = "static_medium"

    # dataset_name = "dynamic1"
    # dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    for DATASET_NAME in DATASET_NAMES[0:]:
        print(f"{DATASETS_PATH/DATASET_NAME}:")
        DATASET_PATH = DATASETS_PATH / "test" / DATASET_NAME
        __refresh_dir(DATASET_PATH / "output")
        run_inference(DATASET_PATH, CosyPose, object_dataset)

if __name__ == "__main__":
    main()
