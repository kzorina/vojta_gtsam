# Standard Library
import json
import os
import gc
########################
# Add cosypose to my path -> dirty
import time
from pathlib import Path
from typing import List, Tuple, Union

# import cosypose
import cv2

# Third Party
import numpy as np
import torch
from PIL import Image
import pickle

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper, 
)


from happypose.pose_estimators.megapose.inference.detector import Detector

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    ObservationTensor,
)
from happypose.toolbox.lib3d.transform import Transform

# HappyPose
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.logging import get_logger, set_logging_level

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



# METHOD_NAME = 'cosy'
METHOD_NAME = 'mega'

DS_NAME = "ycbv"
# DS_NAME = "hope"
OBJECT_NAMES = YCBV_OBJECT_NAMES if DS_NAME == 'ycbv' else HOPE_OBJECT_NAMES
COMMENT = '0.0_threshold'
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

def rendering(predictions, dataset_dir, renderer, K=None):
    labels = predictions.infos["label"]
    # idxs = predictions.infos["coarse_instance_idx"]
    camera_data = CameraData
    camera_data.K = K
    camera_data.TWC = Transform(np.eye(4))
    # Data necessary for image rendering
    object_datas = []
    # for idx, rough_label in zip(idxs, labels):
    for idx, rough_label in enumerate(labels):
        pred = predictions.poses[idx].numpy()
        # label = OBJECT_NAMES[rough_label.split("-")[1]]  # (KZ) was here before... for hope? 
        label = rough_label.split("-")[1]
        object_datas.append(ObjectData(label=label, TWO=Transform(pred)))
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = len(labels) * [
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
    # TCO = predictions.poses
    # K = torch.from_numpy(K)
    # K = K.unsqueeze(0)
    # K = K.repeat(TCO.shape[0], 1, 1)
    
    # renderings = renderer.render(
    #     labels=labels,
    #     TCO=TCO,
    #     K=K,
    #     light_datas=light_datas,
    #     resolution=(height, width),
    #     render_normals=True,
    #     render_depth=True,
    #     render_binary_mask=True,
    # )

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

def predictions_to_dict(predictions, detections):
    entry = {}
    entry_px_count = {}
    poses_tensor: torch.Tensor = predictions.tensors["poses"]
    for idx, label in enumerate(predictions.infos["label"]):
        pose = poses_tensor[idx].numpy()
        px_count = int(detections.masks[idx].sum().cpu())
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


def run_inference(dataset_dir: Path, detector: Detector, pose_estimator) -> None:
    DATASET_NAME = int(dataset_dir.name)

    img_names = sorted(os.listdir(dataset_dir / "rgb"))
    
    scene_camera = get_scene_camera(dataset_dir/"scene_camera.json")

    start_time = time.time()
    all_predictions = []
    all_px_counts = []
    all_tensor_predictions = []
    for i in range(len(img_names)):
        K = scene_camera[i]["cam_K"]

        img_name = img_names[i]
        frame_processing_time_start = time.time()
        img_path = dataset_dir / "rgb" / img_name
        rgb, depth, camera_data = load_observation(dataset_dir, img_path, K=K)
        observation = data_to_observation(rgb, depth, camera_data)

        # Common API between cosypose and megapose pose estimator
        # detections = detector.get_detections(observation, output_masks=True, detection_th=0.7)
        detections = detector.get_detections(observation, output_masks=True)
        final_preds, all_preds = pose_estimator.run_inference_pipeline(observation, detections)
        predictions = final_preds.cpu()
        all_tensor_predictions.append(predictions.clone())
        if i == 0:
            mesh_units = 'mm'
            rigid_objects = []
            for obj_id in YCBV_OBJECT_NAMES.keys():
                mesh_path = dataset_dir.parent.parent / f"models/{obj_id}.ply"
                rigid_objects.append(RigidObject(label=obj_id, mesh_path=mesh_path, mesh_units=mesh_units))
            # rigid_objects = [RigidObject(label=obj_id, mesh_path=f"/home/ros/kzorina/vojtas/ycbv/test/000048/models/{obj_id}.ply", mesh_units=mesh_units) for obj_id in YCBV_OBJECT_NAMES.keys()]
            # TODO: fix mesh units (old TODO)
            rigid_object_dataset = RigidObjectDataset(rigid_objects)
            renderer = Panda3dSceneRenderer(rigid_object_dataset)
            renderings = rendering(predictions, dataset_dir, renderer, K=K)

            # save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, renderings.rgb, predictions.tensors['boxes_crop'].data.numpy())
            save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, renderings.rgb, predictions.tensors['boxes_rend'].data.numpy())

            masks = detections.masks.sum(0).cpu().numpy().astype(np.uint8)*20
            masks_rgb = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)
            save_prediction_img(
                dataset_dir / "output_mask", img_name, rgb, masks_rgb, predictions.tensors['boxes_rend'].data.numpy())

        # if i == 0:
        #     renderer = pose_estimator.refiner_model.renderer
        #     renderings = rendering(predictions, dataset_dir, renderer, K, rgb.shape)

        #     # save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, renderings.rgb, predictions.tensors['boxes_crop'].data.numpy())
        #     save_prediction_img(dataset_dir / "output_bbox", img_name, rgb, renderings.rgbs, predictions.tensors['boxes_rend'].data.numpy())

        #     masks = detections.masks.sum(0).cpu().numpy().astype(np.uint8)*20
        #     masks_rgb = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)
        #     save_prediction_img(dataset_dir / "output_mask", img_name, rgb, masks_rgb, predictions.tensors['boxes_rend'].data.numpy())
        
        poses, bboxes = predictions_to_dict(predictions, detections)
        # breakpoint()
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
    # print("PREDICTIONS NOT SAVED!!! REMOVE A COMMENT!!!")
    save_preditions_data(dataset_dir / f"{METHOD_NAME}_{COMMENT}_frames_prediction.p", all_predictions)
    save_preditions_data(dataset_dir / f"{METHOD_NAME}_{COMMENT}_frames_px_counts.p", all_px_counts)
    save_preditions_data(dataset_dir / f"{METHOD_NAME}_{COMMENT}_tensor_predictions.p", all_tensor_predictions)
    # method_dataset-subset_blabla
    export_bop(convert_frames_to_bop({DATASET_NAME: all_predictions}), dataset_dir / f"{METHOD_NAME}_{DS_NAME}-test_{COMMENT}_frames_prediction.csv")


def main():
    #  in case of OOM error try this:
    #  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
    # torch.cuda.empty_cache()
    set_logging_level("info")
    # export HAPPYPOSE_DATA_DIR=/home/ros/sandbox_mf/data/local_data_happypose
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthStatic")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthTest")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthDynamic")
    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/SynthDynamicOcclusion")
    DATASETS_PATH = Path("/home/ros/kzorina/vojtas/ycbv")

    # DS_NAME used to load the correct object models and detector/pose estimators weights

  
    # DATASET_NAMES = ["000048"]
    DATASET_NAMES = ["000048", "000049", "000050", "000051", "000052", "000053", "000054", "000055", "000056", "000057", "000058", "000059"]  # ycbv
    # DATASET_NAMES = ["000048", "000049", "000050", "000051", "000052"]
    # DATASET_NAMES = ["000058", "000059"]
    # DATASET_NAMES = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009"]  # hope, synth

    # DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets/ycbv")
    MESHES_PATH = DATASETS_PATH/"meshes"

    from happypose.toolbox.datasets.datasets_cfg import make_object_dataset
    object_dataset = make_object_dataset(DS_NAME)

    if METHOD_NAME == 'cosy':
        CosyPose = CosyPoseWrapper(dataset_name=DS_NAME, n_workers=0, renderer_type='bullet', load_synth_real=True)
        detector = CosyPose.detector
        pose_estimator = CosyPose.pose_predictor
    elif METHOD_NAME == 'mega':
        from happypose.toolbox.utils.load_model import load_named_model
        from happypose.toolbox.inference.utils import load_detector
        DETECTOR_RUN_IDS = {
            'hope': "detector-bop-hope-pbr--15246",
            'ycbv': "detector-bop-ycbv-pbr--970850",
        }
        detector = load_detector(run_id=DETECTOR_RUN_IDS[DS_NAME], device=device)
        pose_estimator = load_named_model("megapose-1.0-RGB-multi-hypothesis", object_dataset=object_dataset)
    else:
        raise NotImplementedError(f'Unknown method {METHOD_NAME}')
    # dataset_name = "crackers_new"
    # dataset_name = "static_medium"

    # dataset_name = "dynamic1"
    # dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    for DATASET_NAME in DATASET_NAMES:
        print(f"{DATASETS_PATH/DATASET_NAME}:")
        DATASET_PATH = DATASETS_PATH / "test" / DATASET_NAME
        __refresh_dir(DATASET_PATH / "output")
        run_inference(DATASET_PATH, detector, pose_estimator)
    import csv
    output_file = f'/home/ros/kzorina/vojtas/ycbv/{METHOD_NAME}_{DS_NAME}-test_{COMMENT}_frames_prediction.csv'

    # Merge CSV files
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for i, test_id in enumerate([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]):
            # file_name_v1 = f"/home/ros/kzorina/vojtas/ycbv/test/0000{test_id}/{METHOD_NAME}_{DS_NAME}-test_frames_prediction_{COMMENT}.csv"
            # file_name_v2 = f"/home/ros/kzorina/vojtas/ycbv/test/0000{test_id}/{METHOD_NAME}_{DS_NAME}-test_{COMMENT}_frames_prediction.csv"
            # file_name = file_name_v1 if os.path.exists(file_name_v1) else file_name_v2
            file_name = f"/home/ros/kzorina/vojtas/ycbv/test/0000{test_id}/{METHOD_NAME}_{DS_NAME}-test_{COMMENT}_frames_prediction.csv"
            print('processing ', file_name)
            with open(file_name, 'r') as infile:
                if i != 0:
                    infile.readline()
                reader = csv.reader(infile)
                for row in reader:
                    writer.writerow(row)
    print("saved merged ffile to ", output_file)
    # merge_inferences(DATASET_PATH, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "frames_prediction.p", f'cosypose_{DATASET_NAME}-test.csv', dataset_type)

if __name__ == "__main__":
    main()

# import csv
# file_names = [
#     "/home/ros/kzorina/vojtas/ycbv/test/000048/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000049/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000050/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000051/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000052/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000053/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000054/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000055/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000056/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000057/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000058/cosy_ycbv-test_frames_prediction.csv",
#     "/home/ros/kzorina/vojtas/ycbv/test/000059/cosy_ycbv-test_frames_prediction.csv"
# ]

# import csv
# output_file = '/home/ros/kzorina/vojtas/ycbv/cosy_ycbv-test_frames_prediction.csv'

# # Merge CSV files
# with open(output_file, 'w', newline='') as outfile:
#     writer = csv.writer(outfile)
#     for i, test_id in enumerate([48, 49, 50, 51, 52, 53,54,55,56,57,58,59]):
#         with open(f"/home/ros/kzorina/vojtas/ycbv/test/0000{test_id}/cosy_ycbv-test_frames_prediction.csv", 'r') as infile:
#             if i != 0:
#                 infile.readline()
#             reader = csv.reader(infile)
#             for row in reader:
#                 writer.writerow(row)