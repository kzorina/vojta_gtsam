import rosbag
import numpy as np
import cv2
import os
from pathlib import Path
from collections import defaultdict
import pinocchio as pin

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

YCBV_OBJECT_NAMES = {"ycbv-obj_000001": "01_master_chef_can",
    "ycbv-obj_000002": "02_cracker_box",
    "ycbv-obj_000003": "03_sugar_box",
    "ycbv-obj_000004": "04_tomatoe_soup_can",
    "ycbv-obj_000005": "05_mustard_bottle",
    "ycbv-obj_000006": "06_tuna_fish_can",
    "ycbv-obj_000007": "07_pudding_box",
    "ycbv-obj_000008": "08_gelatin_box",
    "ycbv-obj_000009": "09_potted_meat_can",
    "ycbv-obj_000010": "10_banana",
    "ycbv-obj_000011": "11_pitcher_base",
    "ycbv-obj_000012": "12_bleach_cleanser",
    "ycbv-obj_000013": "13_bowl",
    "ycbv-obj_000014": "14_mug",
    "ycbv-obj_000015": "15_power_drill",
    "ycbv-obj_000016": "16_wood_block",
    "ycbv-obj_000017": "17_scissors",
    "ycbv-obj_000018": "18_large_marker",
    "ycbv-obj_000019": "19_large_clamp",
    "ycbv-obj_000020": "20_extra_large_clamp",
    "ycbv-obj_000021": "21_foam_brick"}

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

# def rendering(predictions, renderer, K, resolution=(640, 480)):
#     camera_data = CameraData
#     camera_data.K = K
#     camera_data.TWC = Transform(np.eye(4))
#     camera_data.resolution = resolution
#     object_datas = []
#     for label in predictions:
#         if label != "Camera":
#             for obj_inst in predictions[label]:
#                 if isinstance(obj_inst, dict):
#                     if obj_inst["valid"] == False:
#                         continue
#                     T_co = obj_inst["T_co"]
#                 else:
#                     T_co = obj_inst
#                 object_datas.append(ObjectData(label=label, TWO=Transform(T_co)))
#     camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
#     light_datas = [
#         Panda3dLightData(
#             light_type="ambient",
#             color=((0.6, 0.6, 0.6, 1)),
#         ),
#     ]
#     renderings = renderer.render_scene(
#         object_datas,
#         [camera_data],
#         light_datas,
#         render_depth=False,
#         render_binary_mask=False,
#         render_normals=False,
#         copy_arrays=True,
#     )[0]
#     return renderings

def rendering(renderer, predictions, K: np.ndarray, resolution: tuple):
    camera_data = CameraData(
        K=K,
        TWC=Transform(np.eye(4)),
        resolution=resolution,
    )
    object_datas = []
    for label in predictions:
        for obj_inst in predictions[label]:
            T_co = obj_inst["T_co"]
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

def posemsg_2_SE3(msg):
    pose_v = np.array([
        msg.position.x,
        msg.position.y,
        msg.position.z,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    ])
    return pin.XYZQUATToSE3(pose_v)

def dets_to_lists(dets):
    T_cos = defaultdict(list)
    px_counts = defaultdict(list)
    for det in dets:
        T_co = posemsg_2_SE3(det.posecov.pose).homogeneous
        obj_label = det.obj_label
        if obj_label in HOPE_OBJECT_NAMES:
            obj_label = HOPE_OBJECT_NAMES[obj_label]
        elif obj_label in YCBV_OBJECT_NAMES:
            obj_label = YCBV_OBJECT_NAMES[obj_label]
        pixel_count = det.mask_pixel_count
        T_cos[obj_label].append({"T_co": T_co})
        px_counts[obj_label].append(pixel_count)
    return T_cos, px_counts

def tracks_to_list(tracks, T_bc):
    T_cos = defaultdict(list)
    for track in tracks:
        T_bo = posemsg_2_SE3(track.pose)
        obj_label = track.label
        if obj_label in HOPE_OBJECT_NAMES:
            obj_label = HOPE_OBJECT_NAMES[obj_label]
        elif obj_label in YCBV_OBJECT_NAMES:
            obj_label = YCBV_OBJECT_NAMES[obj_label]
        T_co = T_bc.inverse() * T_bo
        T_cos[obj_label].append({"T_co":T_co, "id":track.track_id})
    return T_cos

#
# def tracks_to_list(self, tracks, pose_bc: Pose):
#     T_bc = posemsg_2_SE3(pose_bc)
#     T_cos = defaultdict(list)
#     for track in tracks.tracks:
#         T_bo = posemsg_2_SE3(track.pose)
#         obj_label = track.label
#         T_co = T_bc.inverse() * T_bo
#         T_cos[obj_label].append({"T_co": T_co, "id": track.track_id})
#     return T_cos

# /camera/color/camera_info   2281 msgs @  59.5 Hz : sensor_msgs/CameraInfo
# /camera/color/image_raw     2281 msgs @  59.5 Hz : sensor_msgs/Image
# /cosy_detections             516 msgs @  12.9 Hz : ros_cosy_tracking/CosyDetections
# /pose_camera                5884 msgs @ 490.6 Hz : geometry_msgs/PoseStamped
# /tracks

def overlay_images(rgb, render):
    mask = ~(render.sum(axis=-1) == 0)
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = render[mask]

    rgb_overlay = np.zeros_like(render)
    rgb_overlay[~mask] = rgb[~mask] * 0.4 + 255 * 0.6
    rgb_overlay[mask] = render[mask] * 0.9 + 255 * 0.1
    # comparison_img = np.concatenate((rgb, rgb_overlay), axis=1)
    return rgb_overlay

def main():
    OUTPUT_PATH = Path("/media/vojta/Data/video/demo")
    DATASETS_PATH = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "hopeVideo"
    DATASET_NAME = "ycbv"
    MESHES_PATH = DATASETS_PATH/DATASET_NAME/"meshes"
    object_dataset = make_object_dataset(MESHES_PATH)
    renderer = Panda3dSceneRenderer(object_dataset)
    bag = rosbag.Bag('sam_tracking_demo.bag')
    # for topic, msg, t in bag.read_messages(topics=['/cosy_detections']):
    #     pass
    K = None
    resolution = (480, 640)
    last_img = None
    last_camera_info = None
    last_camera_pose = None
    last_cosy_det = None
    last_tracks = None
    last_cosy_render = np.zeros((resolution[0], resolution[1], 3))
    last_tracks_render = np.zeros((resolution[0], resolution[1], 3))
    frame = 0
    video = cv2.VideoWriter(str(OUTPUT_PATH/"overlayed.mp4"), cv2.VideoWriter_fourcc(*'MPEG'), 60.0, (2*resolution[1], resolution[0]))
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/camera/color/camera_info', '/cosy_detections', '/pose_camera', '/tracks']):
        if topic == '/camera/color/image_raw':
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            last_img = cv_image
            cv2.imwrite(str(OUTPUT_PATH / "rgb" / f"{frame:03}.png"), last_img)

            # cv2.imwrite(str(OUTPUT_PATH / "cosy_render" / f"{frame:03}.png"), last_cosy_render)
            # cv2.imwrite(str(OUTPUT_PATH / "tracks_render" / f"{frame:03}.png"), last_tracks_render)
            # cv2.imwrite(str(OUTPUT_PATH / "tracks_render" / f"{frame:03}.png"), last_tracks_render)
            cosy_overlay = overlay_images(last_img, last_cosy_render)
            tracks_overlay = overlay_images(last_img, last_tracks_render)
            combined = np.concatenate((cosy_overlay, tracks_overlay), axis=1).astype(np.uint8)
            # combined = cv2.cvtColor(combined, cv2.COLOR_HSV2BGR)
            # cv2.imwrite(str(OUTPUT_PATH / "overlayed" / f"{frame:03}.png"), combined)

            video.write(combined)

            frame += 1
            print(f"\r{frame}", end='')
            # cv2.imshow("Input", last_img)
            # cv2.waitKey(0)
        elif topic == '/camera/color/camera_info':
            # last_camera_info = msg
            K = np.array(msg.K).reshape((3, 3))
            resolution = (msg.height, msg.width)
            # resolution
        elif topic == '/cosy_detections':
            last_cosy_det, px_counts = dets_to_lists(msg.dets)

            if last_img is not None and last_cosy_det is not None:
                render_rgb = rendering(renderer, last_cosy_det, K, resolution).rgb
                render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_BGR2RGB)
                last_cosy_render = render_rgb
        elif topic == '/pose_camera':
            last_camera_pose = posemsg_2_SE3(msg.pose)
        elif topic == '/tracks':
            last_tracks = tracks_to_list(msg.tracks, last_camera_pose)
            if last_img is not None and last_tracks is not None:
                render_rgb = rendering(renderer, last_tracks, K, resolution).rgb
                render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_BGR2RGB)
                last_tracks_render = render_rgb
                # cv2.imshow("Input", last_tracks_render)
                # cv2.waitKey(0)
    video.release()




    bag.close()

if __name__ == "__main__":
    main()
