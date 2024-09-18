from pathlib import Path
import pickle
import cv2
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import os
import shutil
from collections import defaultdict
from skimage.color import rgb2hsv, hsv2rgb

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.lib3d.transform import Transform
import time

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

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)
# obj_label = '04_tomatoe_soup_can'
# obj_label = '05_mustard_bottle'
# obj_label = '02_cracker_box'
# cosy_gtsam_palette = (((40, 39, 214), (44, 160, 44)),
#                       )
# cosy_gtsam_palette_hsv = (((255, 209, 211), (85, 185, 160)),
#                           ((20, 209, 211), (105, 185, 160)),
#                           ((40, 209, 211), (125, 185, 160)),
#                           ((60, 209, 211), (145, 185, 160)),
#                             ((255, 209, 211), (85, 185, 160)))

# color_map = defaultdict(lambda : ((0, 0, 0), (0, 0, 0)))
# color_map['02_cracker_box'] = ((214, 39, 40), (44, 160, 44))
# color_map['05_mustard_bottle'] = ((215, 69, 40), (44, 160, 63))
# color_map['04_tomatoe_soup_can'] = ((215, 98, 40), (44, 160, 83))
# color_map['08_gelatin_box'] = ((215, 128, 40), (44, 160, 102))
# color_map['03_sugar_box'] = ((215, 157, 40), (44, 121, 121))

color_map = defaultdict(lambda : ((214, 39, 40), (44, 160, 44)))

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
def read_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

# def draw_countour()

def render_obj(renderer, obj_label, T_co, K: np.ndarray, resolution: tuple):
    camera_data = CameraData(
        K=K,
        TWC=Transform(np.eye(4)),
        resolution=resolution,
    )
    object_datas = []
    object_datas.append(ObjectData(label=obj_label, TWO=Transform(T_co)))

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


def get_poses_projection(cosy, tracks, camera_poses, obj_label):
    cosy_poses = np.full((len(cosy)), None, dtype=float)
    tracks_poses = np.full((len(tracks)), None, dtype=float)
    for i in range(len(cosy)):
        T_wc = camera_poses[i]
        if obj_label in cosy[i]:
            T_wo_cosy = pin.SE3(cosy[i][obj_label][0]["T_wo"])
            # dist = np.linalg.norm(T_wo_cosy.translation) + np.linalg.norm(pin.log3(T_wo_cosy.rotation)) * rot_weight
            dist = np.linalg.norm(pin.log3(T_wo_cosy.rotation))
            cosy_poses[i] = dist
        if obj_label in tracks[i]:
            T_co_tracks = pin.SE3(tracks[i][obj_label][0]["T_co"])
            T_wo_tracks = T_wc * T_co_tracks
            # dist = np.linalg.norm(T_wo_tracks.translation) + np.linalg.norm(pin.log3(T_wo_tracks.rotation)) * rot_weight
            dist = np.linalg.norm(pin.log3(T_wo_tracks.rotation))
            tracks_poses[i] = dist
    return cosy_poses, tracks_poses

def get_labels(cosy):
    all_labels = defaultdict(lambda : 0)
    for i in range(len(cosy)):
        # if not cosy[i]:
        for label in cosy[i]:
            all_labels[label] += 1
    obj_labels = []
    for label in all_labels:
        if all_labels[label] > 5:
            obj_labels.append(label)
    return obj_labels

def plot_evolition(ax, cosy_poses, tracks_poses, title, obj_label,  interval=(850, 1200), special_frames=[], width=100, ylim=(0, 3.2)):
    fps=60
    ax.scatter(np.arange(interval[0], interval[1])/fps, cosy_poses[interval[0]:interval[1]], color='tab:red')
    ax.scatter(np.arange(interval[0], interval[1])/fps, tracks_poses[interval[0]:interval[1]], color='tab:green')
    for frame in special_frames:
        ax.scatter([frame/fps], cosy_poses[frame], color='tab:red', marker='o', s=300, linewidths=3, facecolors='none')
        ax.scatter([frame/fps], tracks_poses[frame], color='tab:green', marker='o', s=300, linewidths=3, facecolors='none')

    if title is None:
        ax.set_title(f"{obj_label}")
    else:
        ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_xlim(max(interval[0]/fps + 0.5, 0), max(interval[1], interval[0] + width)/fps)
    ax.set_xlabel(f"time [s]", fontsize=12)
    ax.set_ylabel(f"angular distance [rad]", fontsize=12)
    minor_ticks = np.arange(int(max(interval[0]/fps + 0.5, 0)), int(max(interval[1], interval[0] + width)/fps) + 1, 0.5)

    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_axisbelow(True)
    plt.grid()

# '02_cracker_box'
# '03_sugar_box'
# '05_mustard_bottle'
# '04_tomatoe_soup_can'

def draw_contours(rgb, renders, colors):
    for i, render in enumerate(renders):
        # mask = (~(render.sum(axis=-1) == 0)).astype(np.uint8)
        mask = render

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, colors[i], 8 - i*3)
    return rgb

def merge_images_to_video(input_folder, output_path, fps=60.0):
    files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))
    img = cv2.imread(str(input_folder/files[0]), cv2.IMREAD_COLOR)
    video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MPEG'), fps, (img.shape[1], img.shape[0]))
    for i in range(len(files)):
        file = files[i]
        img = cv2.imread(str(input_folder/file), cv2.IMREAD_COLOR)
        video.write(img)
        print(f"\r({i}/{len(files)})", end='')
    video.release()


def draw_contour_overlays(cosy, tracks, camera_poses, obj_labels, INPUT_PATH, resolution, K, renderer):
    print("rendering contour overlays...")
    for frame in range(len(cosy)):
        rgb = draw_countour_overlay(cosy, tracks, camera_poses, obj_labels, INPUT_PATH, resolution, K, renderer, frame)
        cv2.imwrite(str(INPUT_PATH/"contour_overlay"/f"{frame}.png"), rgb)
        print(f"\r({frame}/{len(cosy)})", end='')
    print("")

def draw_countour_overlay(cosy, tracks, camera_poses, obj_labels, INPUT_PATH, resolution, K, renderer, frame):
    T_wc = camera_poses[frame]
    rgb = cv2.imread(str(INPUT_PATH/"rgb"/f"{frame:03}.png"), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    for obj_label in obj_labels:
        render_cosy = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
        render_tracks = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
        if obj_label in cosy[frame] and len(cosy[frame][obj_label]) > 0:
            for i in range(len(cosy[frame][obj_label])):
                T_co_cosy = T_wc.inverse() * cosy[frame][obj_label][i]['T_wo']
                render_cosy = render_cosy | (~(render_obj(renderer, obj_label, T_co_cosy, K, resolution).rgb.sum(axis=-1) == 0)).astype(np.uint8)
        # if obj_label in tracks[frame] and len(tracks[frame][obj_label]) > 0:
        #     for i in range(len(tracks[frame][obj_label])):
        #         T_co_tracks = tracks[frame][obj_label][i]['T_co']
        #         render_tracks = render_tracks | (~(render_obj(renderer, obj_label, T_co_tracks, K, resolution).rgb.sum(axis=-1) == 0)).astype(np.uint8)
                # render_tracks = render_obj(renderer, obj_label, T_co_tracks, K, resolution).rgb
        rgb = draw_contours(rgb, [render_cosy, render_tracks], color_map[obj_label])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # draw_contours(rgb, [render_tracks], [(0, 255, 0)])
    return rgb

def render_evolution_video(cosy, cosy_poses, tracks_poses, camera_poses, obj_label, INPUT_PATH, output_dir):
    width = 200
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.1)
    start_time = time.time()
    for i in range(len(cosy)):
        if i ==200:
            print(time.time() - start_time, "\n")
        ax.clear()
        plot_evolition(ax, cosy_poses, tracks_poses, None, obj_label, interval=(max(0, i - width), i), width=width)
        plt.savefig(INPUT_PATH / output_dir / f'{i}.png')
        print(f"\r({i}/{len(cosy)})", end='')

def render_evolution_video_fast(cosy, cosy_poses, tracks_poses, camera_poses, obj_label, INPUT_PATH, output_dir):
    width = 100
    fps = 60
    fig, ax = plt.subplots()

    plt.title(f"{obj_label}")
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.1)

    gtsam_x = np.arange(0, width)/fps
    gtsam_y = np.full((width), None)

    cosy_x = np.arange(0, width) / fps
    cosy_y = np.full((width), None)

    line_cosy = ax.plot(cosy_x, cosy_y, color=np.array(color_map[obj_label][0])/255, marker='.', linestyle='None', markersize=15)[0]
    line_gtsam = ax.plot(gtsam_x, gtsam_y, color=np.array(color_map[obj_label][1])/255,  marker='.', linestyle='None', markersize=10)[0]

    ax.set_ylim((0, 3.2))

    ax.set_xlabel(f"time [s]", fontsize=12)
    ax.set_ylabel(f"angular distance [rad]", fontsize=12)

    ax.grid(which='minor', alpha=1)
    plt.grid()
    start_time = time.time()

    def animate(i):
        interval = (max(0, i - width), i)
        gtsam_x = np.arange(interval[0], interval[1]) / fps
        gtsam_y = tracks_poses[interval[0]:interval[1]]

        cosy_x = np.arange(interval[0], interval[1]) / fps
        cosy_y = cosy_poses[interval[0]:interval[1]]
        ax.set_xlim(max(interval[0] / fps, 0), max(interval[1], interval[0] + width) / fps)
        line_cosy.set_xdata(cosy_x)
        line_cosy.set_ydata(cosy_y)
        line_gtsam.set_xdata(gtsam_x)
        line_gtsam.set_ydata(gtsam_y)
        # plt.savefig(INPUT_PATH / output_dir / f'{i}.png')
        print(f"\r({i}/{len(cosy)})", end='')
        return (line_gtsam, line_cosy)

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(cosy), interval=1000/fps)
    ani.save(filename= INPUT_PATH / f"{obj_label}.mp4", writer="ffmpeg", fps=60, dpi=200)
    # plt.show()
def main():
    # INPUT_PATH = Path("/media/vojta/Data/video/demo_dynamic3")

    DATASETS_PATH = Path("/mnt/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "ycbv"
    MESHES_PATH = DATASETS_PATH/DATASET_NAME/"meshes"

    # INPUT_PATH = Path("/mnt/Data/video/demo_tracking")
    INPUT_PATH = Path("/mnt/Data/video/demo_static")
    # INPUT_PATH = Path("/mnt/Data/video/demo_dynamic1")
    # INPUT_PATH = Path("/mnt/Data/video/demo_dynamic2")
    # INPUT_PATH = Path("/mnt/Data/video/demo_dynamic3")
    cosy = read_pickle(INPUT_PATH/"cosy.p")
    tracks = read_pickle(INPUT_PATH/"tracks.p")
    camera_poses = read_pickle(INPUT_PATH/"camera_poses.p")
    # __refresh_dir(INPUT_PATH / "evolution_plots")

    obj_labels = get_labels(cosy)

    K = np.array(((615.15295, 0, 324.57507),
                  (0, 615.24524, 237.81766),
                  (0, 0, 1)))
    resolution = (480, 640)
    object_dataset = make_object_dataset(MESHES_PATH)
    renderer = Panda3dSceneRenderer(object_dataset)
    # frames_to_render = [687, 750, 846]


    # obj_label = '04_tomatoe_soup_can'
    # obj_label = '05_mustard_bottle'
    # obj_label = '02_cracker_box'

    # __refresh_dir(INPUT_PATH / "contour_overlay")
    # draw_contour_overlays(cosy, tracks, camera_poses, obj_labels, INPUT_PATH, resolution, K, renderer)
    # merge_images_to_video(INPUT_PATH/"contour_overlay", INPUT_PATH/f"{INPUT_PATH.name}_contour_overlay.mp4")

    # print("\nrendering pose evolutions...")
    # for obj_label in obj_labels:
    #     output_dir = obj_label
    #     # __refresh_dir(INPUT_PATH / output_dir)
    #     # render_evolution_video(cosy, cosy_poses, tracks_poses, camera_poses, obj_label, INPUT_PATH, output_dir)
    #     cosy_poses, tracks_poses = get_poses_projection(cosy, tracks, camera_poses, obj_label)
    #     render_evolution_video_fast(cosy, cosy_poses, tracks_poses, camera_poses, obj_label, INPUT_PATH, output_dir)
    #     print("\n")

    # obj_label = '02_cracker_box'
    obj_label = '05_mustard_bottle'
    cosy_poses, tracks_poses = get_poses_projection(cosy, tracks, camera_poses, obj_label)
    frames_to_render = [906, 1112, 1168]
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.15)
    plot_evolition(ax, cosy_poses, tracks_poses, "", obj_label, interval=(900, 1180), width=2, ylim=(1.25, 3.5), special_frames=frames_to_render)
    plt.savefig(INPUT_PATH / f'teaser_fig2_red.svg')
    plt.savefig(INPUT_PATH / f'teaser_fig2_red.pdf')
    plt.savefig(INPUT_PATH / f'teaser_fig2_red.png')
    plt.show()
    # for frame in frames_to_render:
    #     rgb = draw_countour_overlay(cosy, tracks, camera_poses, [obj_label], INPUT_PATH, resolution, K, renderer, frame)
    #     cv2.imwrite(str(INPUT_PATH / f"{frame}_red.png"), rgb)
    #     print(f"{frame}")

if __name__ == "__main__":
    main()