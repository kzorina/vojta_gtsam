from pathlib import Path
import os
import shutil
import json
from PIL import Image
import numpy as np

OBJ_IDS = {"AlphabetSoup":1,
           "BBQSauce":2,
           "Butter":3,
           "Cherries":4,
           "ChocolatePudding":5,
           "Cookies":6,
           "Corn":7,
           "CreamCheese":8,
           "GranolaBars":9,
           "GreenBeans":10,
           "Ketchup":11,
           "MacaroniAndCheese":12,
           "Mayo":13,
           "Milk":14,
           "Mushrooms":15,
           "Mustard":16,
           "OrangeJuice":17,
           "Parmesan":18,
           "Peaches":19,
           "PeasAndCarrots":20,
           "Pineapple":21,
           "Popcorn":22,
           "Raisins":23,
           "SaladDressing":24,
           "Spaghetti":25,
           "TomatoSauce":26,
           "Tuna":27,
           "Yogurt":28}

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

def __soft_refresh_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def __load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def __export_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file)

def main():
    DATASETS_DIR = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    DATASET_NAME = "hope_video"
    dataset_path = DATASETS_DIR/DATASET_NAME
    __soft_refresh_dir(dataset_path)
    __soft_refresh_dir(dataset_path/"test")
    SOURCE_PATH = Path("hope_video")
    scene_names = sorted(os.listdir(SOURCE_PATH))
    for scene_name in scene_names:
        scene_out_path = dataset_path/"test"/f"{int(scene_name[6:]):06d}"
        __soft_refresh_dir(scene_out_path)
        __soft_refresh_dir(scene_out_path/"rgb")
        __soft_refresh_dir(scene_out_path/"depth")
        scene_in_path = SOURCE_PATH/scene_name
        file_names = sorted(os.listdir(scene_in_path))
        scene_camera = {}
        scene_gt = {}
        for i in range((len(file_names) - 1)//3):
            name = f"{i:04d}"
            rgb = Image.open(scene_in_path/f"{name}_rgb.jpg")
            rgb.save(scene_out_path/"rgb"/f"{i+1:06d}.png")
            shutil.copyfile(scene_in_path/f"{name}_depth.png", scene_out_path/"depth"/f"{i+1:06d}.png")
            gt_data = __load_json(scene_in_path / f"{name}.json")
            pass
            K = np.array(gt_data["camera"]["intrinsics"])
            T_cw = np.array(gt_data["camera"]["extrinsics"])
            scene_camera[f"{i+1}"] = {"cam_K":K.flatten().tolist(), "cam_R_w2c":T_cw[:3, :3].flatten().tolist(), "cam_t_w2c":(T_cw[:3, 3].flatten()*1000).tolist(), "depth_scale":123}
            scene_gt[f"{i+1}"] = []
            for object in gt_data["objects"]:
                T_cm = np.array(object["pose"])
                obj_name = object["class"]
                scene_gt[f"{i+1}"].append({"cam_R_m2c":T_cm[:3, :3].flatten().tolist(), "cam_t_m2c":(T_cm[:3, 3].flatten()*10).tolist(), "obj_id":OBJ_IDS[obj_name]})
        __export_json(scene_camera, scene_out_path/"scene_camera.json")
        __export_json(scene_gt, scene_out_path/"scene_gt.json")
        print(f"{scene_name} done")


    pass

if __name__ == "__main__":
    main()