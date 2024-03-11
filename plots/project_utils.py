#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-29
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import json

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from matplotlib import colors
import trimesh


def project_object_to_mask(obj, camera_matrix, pose, im_size, scale=1e-3):
    R = pose[:3, :3]
    t = pose[:3, 3]
    points = trimesh.sample.sample_surface_even(obj, 100000)[0]
    proj = camera_matrix @ (scale * R @ points.T + t[:, None])
    proj = proj[:2] / proj[2]
    proj = proj.astype(int)
    proj = proj[
        :,
        (proj[0] >= 0)
        & (proj[0] < im_size[1])
        & (proj[1] >= 0)
        & (proj[1] < im_size[0]),
    ]
    proj_mask = np.zeros(im_size, dtype=bool)
    proj_mask[proj[1], proj[0]] = True
    binary_fill_holes(proj_mask, output=proj_mask, structure=np.ones((3, 3)))
    return proj_mask


def mask_to_rgba(mask, color="tab:red"):
    """Convert mask to RGBA image."""
    rgb = colors.to_rgb(color)
    out = np.ones((*mask.shape, 4))
    out[:, :, :3] = rgb
    out[~mask, 3] = 0
    return out


def get_contour(mask, iterations=5, color="tab:red"):
    """Get contour of the mask as RGBA image. use erosion for negative number of
    iterations and dilation for positive."""
    rgb = colors.to_rgb(color)
    out = np.ones((*mask.shape, 4))
    out[:, :, :3] = rgb
    if iterations >= 0:
        contour = binary_dilation(mask, iterations=iterations) ^ mask
    else:
        contour = binary_erosion(mask, iterations=-iterations) ^ mask
    out[~contour, 3] = 0
    return out


def camera_matrix_from_dataset(ds_path):
    camera_json = json.load(open(ds_path / "camera.json"))
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = camera_json["fx"]
    camera_matrix[1, 1] = camera_json["fy"]
    camera_matrix[0, 2] = camera_json["cx"]
    camera_matrix[1, 2] = camera_json["cy"]
    return camera_matrix
