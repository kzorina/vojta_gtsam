from pathlib import Path
import json
import os
import numpy

def load_scene_obj_ids(path):
    with open(path) as json_file:
        data:dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = []
        frame = i+1
        for object in data[str(frame)]:
            obj_id = object["obj_id"]
            entry.append(obj_id)
        parsed_data.append(entry)
    return parsed_data

def __export_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=1)
def main():
    DATASETS_DIR = Path("/media/vojta/Data/HappyPose_Data/bop_datasets")
    # DATASET_NAME = "hopeVideo"
    DATASET_NAME = "SynthStatic"
    dataset_path = DATASETS_DIR / DATASET_NAME
    scene_names = sorted(os.listdir(DATASETS_DIR / DATASET_NAME / "test"))
    output = []
    for scene_name in scene_names:
        scene_path = dataset_path / "test" / scene_name
        scene_obj_ids = load_scene_obj_ids(scene_path / "scene_gt.json")
        for im_id in range(0, len(scene_obj_ids), 15):
            for obj_id in scene_obj_ids[im_id]:
                entry = {"im_id":im_id + 1, "inst_count":1, "obj_id":obj_id, "scene_id": int(scene_name)}
                output.append(entry)
    __export_json(output, dataset_path/"test_targets_bop19.json")


if __name__ == "__main__":
    main()