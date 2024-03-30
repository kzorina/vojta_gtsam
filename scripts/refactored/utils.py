import json
import numpy as np
from pathlib import Path
import pickle
import cov

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