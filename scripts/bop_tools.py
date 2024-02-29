

import pickle
import csv
from pathlib import Path

YCBV_OBJECT_IDS = {"01_master_chef_can":1,
    "02_cracker_box":2,
    "03_sugar_box":3,
    "04_tomatoe_soup_can":4,
    "05_mustard_bottle":5,
    "06_tuna_fish_can":6,
    "07_pudding_box":7,
   "08_gelatin_box":8,
    "09_potted_meat_can":9,
    "10_banana":10,
    "11_pitcher_base":11,
    "12_bleach_cleanser":12,
    "13_bowl":13,
    "14_mug":14,
    "15_power_drill":15,
    "16_wood_block":16,
    "17_scissors":17,
    "18_large_marker":18,
    "19_large_clamp":19,
    "20_extra_large_clamp":20,
    "21_foam_brick":21}

HOPE_OBJECT_IDS = {"AlphabetSoup":1,
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

OBJECT_IDS = {"ycbv": YCBV_OBJECT_IDS,
              "hope": HOPE_OBJECT_IDS}
def load_data(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def convert_frames_to_bop(frames: dict[[dict]], dataset_name="ycbv", translate_obj_ids=True) -> [dict]:
    """
    :param frames: {"[scene_id]": [{"object_name": [T_co, T_co, T_co...]}]}
    :returns [{"scene_id":[scene_id],
               "im_id":...,
               "obj_id":...,
               "score":...,
               "R":...,
               "t":...,
               "time":...}]
    """
    output = []

    for scene_id in frames:
        for im_id in range(len(frames[scene_id])):
            for obj_id in frames[scene_id][im_id]:
                for object in frames[scene_id][im_id][obj_id]:
                    if isinstance(object, dict):
                        if not object["valid"]:
                            continue
                        T_co = object["T_co"]
                    else:
                        T_co = object
                    R = " ".join(list(map(str, T_co[:3, :3].flatten().tolist())))
                    t = " ".join(list(map(str, (T_co[:3, 3]*1000).flatten().tolist())))
                    time = -1
                    score = 1
                    if translate_obj_ids:
                        final_obj_id = OBJECT_IDS[dataset_name][obj_id]
                    else:
                        final_obj_id = obj_id
                    entry = {"scene_id":scene_id, "im_id":im_id + 1, "obj_id":final_obj_id, "score":score, "R":R, "t":t, "time":time}
                    output.append(entry)
    return output

def export_bop(bop_entries, path:Path):
    header = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    with open(path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter = ',', fieldnames=header)
        writer.writeheader()
        writer.writerows(bop_entries)
def main():
    dataset_name = "crackers_duplicates"
    dataset_path = Path(__file__).parent.parent / "datasets" / dataset_name
    export_path = dataset_path/"bop_inference"
    frames_dict = load_data(dataset_path/"frames_prediction.p")
    bop_frames = convert_frames_to_bop({"bagr": frames_dict})
    export_bop(bop_frames, export_path)

if __name__ == "__main__":
    main()