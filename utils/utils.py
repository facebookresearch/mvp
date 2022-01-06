# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Miscellaneous utilities"""
import os
import importlib
import re
import sys

import numpy as np

import torch

def findmaxiters(path):
    iternum = 0
    with open(path, "r") as f:
        for line in f.readlines():
            match = re.search("Iteration (\d+).* ", line)
            if match is not None:
                it = int(match.group(1))
                if it > iternum:
                    iternum = it
    return iternum

class Logger(object):
    """Duplicates all stdout to a file."""
    def __init__(self, path, mode):
        if mode == "w" and os.path.exists(path):
            print(path + " exists")
            sys.exit(0)

        self.log = open(path, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def tocuda(d):
    if isinstance(d, torch.Tensor):
        return d.to("cuda", non_blocking=True)
    elif isinstance(d, dict):
        return {k: tocuda(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tocuda(v) for v in d]
    else:
        return d

def findbatchsize(d):
    if isinstance(d, torch.Tensor):
        return d.size(0)
    elif isinstance(d, dict):
        return findbatchsize(next(iter(d.values())))
    elif isinstance(d, list):
        return len(d)
    else:
        return None

def load_obj(path):
    """Load wavefront OBJ from file."""
    v = []
    vt = []
    vindices = []
    vtindices = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])
                if line.find("/") != -1:
                    vtindices.append([int(entry.split('/')[1]) - 1 for entry in line.split()[1:]])

    return v, vt, vindices, vtindices

def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters for cameras.
    
    KRT file is a text file with 1 or more entries like:
        <camera name> <image width (pixels)> <image height (pixels)>
        <f_x> 0 <c_x>
        0 <f_y> <c_y>
        0 0 1
        <k_1> <k_2> <p_1> <p_2> <k_3>
        <r_11> <r_12> <r_13> <t_x>
        <r_21> <r_22> <r_23> <t_y>
        <r_31> <r_32> <r_33> <t_z>
        [blank line]

    The parameters are in OpenCV format described here:
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

    Note that the code assumes undistorted images as input, so the distortion
    coefficients are ignored."""
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            namesplit = name.split()
            if len(namesplit) > 1:
                name, width, height = namesplit[0], namesplit[1], namesplit[2]
                size = {"size": np.array([width, height])}
            else:
                name = namesplit[0]
                size = {}

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin), **size}

    return cameras

def downsample(image, factor):
    if factor == 1:
        return image
    import scipy.ndimage
    image = image.copy('C')
    for i in range(image.shape[-1]):
        image[..., i] = scipy.ndimage.filters.gaussian_filter(image[..., i], 2. * factor / 6)
    return image[::factor, ::factor]

def downsample_mask(mask, factor):
    resultheight = mask.shape[0] // factor
    resultwidth = mask.shape[1] // factor
    result = np.ones((resultheight, resultwidth, mask.shape[2]), dtype=mask.dtype)
    for i in range(factor):
        for j in range(factor):
            result &= mask[i::factor, j::factor]
    return result

def gentritex(v, vt, vi, vti, texsize):
    """Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates"""
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
            (np.arange(texsize) + 0.5) / texsize,
            (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0]))*texsize)+2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1]))*texsize)+2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((baryu, baryv, baryw), axis=-1),
                    baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                    viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                    vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return viim, vtiim, baryim
