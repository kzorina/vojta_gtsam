import json

import gtsam
import numpy as np
from pathlib import Path
import pickle
import cov
from collections import defaultdict

HOPE_OBJECT_NAMES = {
    "obj_000001": "AlphabetSoup",
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
    "obj_000028": "Yogurt",
}
OBJ_IDS = {
    "AlphabetSoup": 1,
    "BBQSauce": 2,
    "Butter": 3,
    "Cherries": 4,
    "ChocolatePudding": 5,
    "Cookies": 6,
    "Corn": 7,
    "CreamCheese": 8,
    "GranolaBars": 9,
    "GreenBeans": 10,
    "Ketchup": 11,
    "MacaroniAndCheese": 12,
    "Mayo": 13,
    "Milk": 14,
    "Mushrooms": 15,
    "Mustard": 16,
    "OrangeJuice": 17,
    "Parmesan": 18,
    "Peaches": 19,
    "PeasAndCarrots": 20,
    "Pineapple": 21,
    "Popcorn": 22,
    "Raisins": 23,
    "SaladDressing": 24,
    "Spaghetti": 25,
    "TomatoSauce": 26,
    "Tuna": 27,
    "Yogurt": 28,
}

def load_pickle(path: Path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_camera(path):
    """
    loads scene_camera.json from a BOP dataset into a dictionary
    :param path:
    :return:
    """
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

def load_scene_gt(path, label_list=None):
    with open(path) as json_file:
        data: dict = json.load(json_file)
    parsed_data = []
    for i in range(len(data)):
        entry = defaultdict(lambda: [])
        frame = i + 1
        for object in data[str(frame)]:
            T_cm = np.zeros((4, 4))
            T_cm[:3, :3] = np.array(object["cam_R_m2c"]).reshape((3, 3))
            T_cm[:3, :3] = T_cm[:3, :3]
            T_cm[:3, 3] = np.array(object["cam_t_m2c"]) / 1000
            T_cm[3, 3] = 1
            obj_id = object["obj_id"]
            entry[label_list[obj_id - 1]].append(T_cm)
        parsed_data.append(entry)
    return parsed_data

def merge_T_cos_px_counts(T_cos, px_counts):
    ret = {}
    for obj_label in T_cos:
        assert obj_label in px_counts
        assert len(T_cos[obj_label]) == len(px_counts[obj_label])
        ret[obj_label] = []
        for obj_idx in range(len(T_cos[obj_label])):
            T_co = T_cos[obj_label][obj_idx]
            Q = cov.measurement_covariance(T_co, px_counts[obj_label][obj_idx])
            ret[obj_label].append({"T_co":T_co, "Q":Q})
    return ret

def format_time(t):
    return f"{int(t)//3600}:{(int(t)%3600)//60}:{(t%60):.2f}"

def parse_variable_index(vi:gtsam.VariableIndex):
    entries = vi.__repr__()[:-1].split('\n')
    factor_keyed = defaultdict(list)
    variable_keyed = defaultdict(list)
    for entry in entries[1:]:
        data = entry.split(' ')[1:]
        variable = data[0][:-1]
        factors = data[1:]
        variable_keyed[variable] += factors
        for factor in factors:
            factor_keyed[factor].append(variable)

    return dict(factor_keyed), dict(variable_keyed)
