# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset class for multiview video datasets"""
import itertools
import multiprocessing
import os
from typing import Optional, Callable

import numpy as np

from scipy.ndimage.morphology import binary_dilation

from PIL import Image

import torch.utils.data

import cv2
cv2.setNumThreads(0)

from utils import utils

class ImageLoader:
    def __init__(self, bgpath, blacklevel):
        self.bgpath = bgpath
        self.blacklevel = np.array(blacklevel, dtype=np.float32)

    def __call__(self, cam, size):
        try:
            imagepath = self.bgpath.format(cam)
            # determine downsampling necessary
            image = np.asarray(Image.open(imagepath), dtype=np.uint8)
            if image.shape[0] != size[1]:
                image = utils.downsample(image, image.shape[0] // size[1])
            image = image.transpose((2, 0, 1)).astype(np.float32)
            image = np.clip(image - self.blacklevel[:, None, None], 0., None)
            bgmask = (~binary_dilation(np.mean(image, axis=0) > 128, iterations=16)).astype(np.float32)[None, :, :]
        except:
            print("exception loading bg image", cam)
            image = None
            bgmask = None

        return cam, image, bgmask

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
            krtpath : str,
            geomdir : str,
            imagepath : str,
            keyfilter : list,
            camerafilter : Callable[[str], bool],
            segmentfilter : Callable[[str], bool]=lambda x: True,
            framelist : Optional[list]=None,
            frameexclude : list=[],
            maxframes : int=-1,
            bgpath : Optional[str]=None,
            returnbg : bool=False,
            baseposesegframe : tuple=None,
            baseposepath : Optional[str]=None,
            fixedcameras : list=[],
            fixedframesegframe : Optional[tuple]=None,
            fixedcammean : float=0.,
            fixedcamstd : float=1.,
            fixedcamdownsample : int=4,
            standardizeverts : bool=True,
            standardizeavgtex : bool=True,
            standardizetex : bool=False,
            avgtexsize : int=1024,
            texsize : int=1024,
            subsampletype : Optional[str]=None,
            subsamplesize : int=0,
            downsample : float=1.,
            blacklevel : list=[0., 0., 0.],
            maskbright : bool=False,
            maskbrightbg : bool=False
            ):
        """
        Dataset class for loading synchronized multi-view video (optionally
        with tracked mesh data).

        Parameters
        ----------
        krtpath : str,
            path to KRT file. See utils.utils.load_krt docstring for details
        geomdir : str,
            base path to geometry data (tracked meshes, unwrapped textures,
            rigid transforms)
        imagepath : str,
            path to images. should be a string that accepts "seg", "cam", and
            "frame" format keys (e.g., "data/{seg}/{cam}/{frame}.png")
        keyfilter : list,
            list of items to load and return (e.g., images, textures, vertices)
            available in this dataset:
            'fixedcamimage' -- image from a particular camera (unlike 'image',
                this image will always be from a specified camera)
            'fixedframeimage' -- image from a particular frame and camera
                (always the same)
            'verts' -- tensor of Kx3 vertices
            'tex' -- texture map as (3 x T_h x T_w) tensor
            'avgtex' -- texture map averaged across all cameras
            'modelmatrix' -- rigid transformation at frame
                (relative to 'base' pose)
            'camera' -- camera pose (intrinsic and extrinsic)
            'image' -- camera image as (3 x I_h x I_w) tensor
            'bg' -- background image as (3 x I_h x I_w) tensor
            'pixelcoords' -- pixel coordinates of image to evaluate
                (used to subsample training images to reduce memory usage)
        camerafilter : Callable[[str], bool],
            lambda function for selecting cameras to include in dataset
        segmentfilter : Callable[[str], bool]
            lambda function for selecting segments to include in dataset
            Segments are contiguous sets of frames.
        framelist=None : list[tuple[str, str]],
            list of (segment, framenumber), used instead of segmentfilter
        frameexclude : list[str],
            exclude these frames from being loaded
        maxframes : int,
            maximum number of frames to load
        bgpath : Optional[str],
            path to background images if available
        returnbg : bool,
            True to return bg images in each batch, false to store them
            into self.bg
        baseposesegframe : tuple[str, str]
            segment, frame of headpose to be used as the "base" head pose
            (used for modelmatrix)
        baseposepath : str,
            path to headpose to be used as the "base" head pose, used instead
            of transfseg, transfframe
        fixedcameras : list,
            list of cameras to be returned for 'fixedcamimage'
        fixedframesegframe : tuple[str, str]
            segment and frame to be used for 'fixedframeimage'
        fixedcammean : float,
        fixedcamstd : float,
            norm stats for 'fixedcamimage' and 'fixedframeimage'
        standardizeverts : bool,
            remove mean/std from vertices
        standardizeavgtex : bool,
            remove mean/std from avgtex
        standardizetex : bool,
            remove mean/std from view-dependent texture
        avgtexsize : int,
            average texture map (averaged across viewpoints) dimension
        texsize : int,
            texture map dimension
        subsampletype : Optional[str],
            type of subsampling to do (determines how pixelcoords is generated)
            one of [None, "patch", "random", "random2", "stratified"]
        subsamplesize : int,
            dimension of subsampling
        downsample : float,
            downsample target image by factor
        blacklevel : tuple[float, float, float]
            black level to subtract from camera images
        maskbright : bool,
            True to not include bright pixels in loss
        maskbrightbg : bool,
            True to not include bright background pixels in loss
        """
        # options
        self.keyfilter = keyfilter
        self.fixedcameras = fixedcameras
        self.fixedframesegframe = fixedframesegframe
        self.fixedcammean = fixedcammean
        self.fixedcamstd = fixedcamstd
        self.fixedcamdownsample = fixedcamdownsample
        self.standardizeverts = standardizeverts
        self.standardizeavgtex = standardizeavgtex
        self.standardizetex = standardizetex
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.downsample = downsample
        self.returnbg = returnbg
        self.blacklevel = blacklevel
        self.maskbright = maskbright
        self.maskbrightbg = maskbrightbg

        # compute camera/frame list
        krt = utils.load_krt(krtpath)

        self.allcameras = sorted(list(krt.keys()))
        self.cameras = list(filter(camerafilter, self.allcameras))

        # compute camera positions
        self.campos, self.camrot, self.focal, self.princpt, self.size = {}, {}, {}, {}, {}
        for cam in self.allcameras:
            self.campos[cam] = (-np.dot(krt[cam]['extrin'][:3, :3].T, krt[cam]['extrin'][:3, 3])).astype(np.float32)
            self.camrot[cam] = (krt[cam]['extrin'][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(krt[cam]['intrin'][:2, :2]) / downsample).astype(np.float32)
            self.princpt[cam] = (krt[cam]['intrin'][:2, 2] / downsample).astype(np.float32)
            self.size[cam] = np.floor(krt[cam]['size'].astype(np.float32) / downsample).astype(np.int32)

        # set up paths
        self.imagepath = imagepath
        if geomdir is not None:
            self.vertpath = os.path.join(geomdir, "tracked_mesh", "{seg}", "{frame:06d}.bin")
            self.transfpath = os.path.join(geomdir, "tracked_mesh", "{seg}", "{frame:06d}_transform.txt")
            self.texpath = os.path.join(geomdir, "unwrapped_uv_1024", "{seg}", "{cam}", "{frame:06d}.png")
        else:
            self.transfpath = None

        # build list of frames
        if framelist is None:
            framelist = np.genfromtxt(os.path.join(geomdir, "frame_list.txt"), dtype=np.str)
            self.framelist = [tuple(sf) for sf in framelist if segmentfilter(sf[0]) and sf[1] not in frameexclude]
        else:
            self.framelist = framelist

        # truncate or extend frame list
        if maxframes <= len(self.framelist):
            if maxframes > -1:
                self.framelist = self.framelist[:maxframes]
        else:
            repeats = (maxframes + len(self.framelist) - 1) // len(self.framelist)
            self.framelist = (self.framelist * repeats)[:maxframes]

        # cartesian product with cameras
        self.framecamlist = [(x, cam)
                for x in self.framelist
                for cam in (self.cameras if len(self.cameras) > 0 else [None])]

        # set base pose
        if baseposepath is not None:
            self.basetransf = np.genfromtxt(baseposepath, max_rows=3).astype(np.float32)
        elif baseposesegframe is not None:
            self.basetransf = np.genfromtxt(self.transfpath.format(
                seg=baseposesegframe[0],
                frame=baseposesegframe[1])).astype(np.float32)
        else:
            raise Exception("base transformation must be provided")

        # load normstats
        if "avgtex" in keyfilter or "tex" in keyfilter:
            texmean = np.asarray(Image.open(os.path.join(geomdir, "tex_mean.png")), dtype=np.float32)
            self.texstd = float(np.genfromtxt(os.path.join(geomdir, "tex_var.txt")) ** 0.5)

        if "avgtex" in keyfilter:
            self.avgtexsize = avgtexsize
            avgtexmean = texmean
            if avgtexmean.shape[0] != self.avgtexsize:
                avgtexmean = cv2.resize(avgtexmean, dsize=(self.avgtexsize, self.avgtexsize), interpolation=cv2.INTER_LINEAR)
            self.avgtexmean = avgtexmean.transpose((2, 0, 1)).astype(np.float32).copy("C")

        if "tex" in keyfilter:
            self.texsize = texsize
            if texmean.shape[0] != self.texsize:
                texmean = cv2.resize(texmean, dsize=self.texsize, interpolation=cv2.INTER_LINEAR)
            self.texmean = texmean.transpose((2, 0, 1)).astype(np.float32).copy("C")

        if "verts" in keyfilter:
            self.vertmean = np.fromfile(os.path.join(geomdir, "vert_mean.bin"), dtype=np.float32).reshape((-1, 3))
            self.vertstd = float(np.genfromtxt(os.path.join(geomdir, "vert_var.txt")) ** 0.5)

        # load background images
        if bgpath is not None:
            readpool = multiprocessing.Pool(40)
            reader = ImageLoader(bgpath, blacklevel)
            self.bg = {cam: (image, bgmask)
                    for cam, image, bgmask
                    in readpool.starmap(reader, zip(self.cameras, [self.size[x] for x in self.cameras]))
                    if image is not None}
        else:
            self.bg = {}

    def get_allcameras(self):
        return self.allcameras

    def get_krt(self):
        return {k: {
                "pos": self.campos[k],
                "rot": self.camrot[k],
                "focal": self.focal[k],
                "princpt": self.princpt[k],
                "size": self.size[k]}
                for k in self.allcameras}

    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter and not self.returnbg:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam][0]).to("cuda")

    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        (seg, frame), cam = self.framecamlist[idx]

        result = {}

        result["segid"] = seg
        result["frameid"] = frame
        if cam is not None:
            result["cameraid"] = cam

        validinput = True

        # image from one or more cameras (those cameras are fixed over the dataset)
        if "fixedcamimage" in self.keyfilter:
            ninput = len(self.fixedcameras)

            fixedcamimage = []
            for i in range(ninput):
                imagepath = self.imagepath.format(seg=seg, cam=self.fixedcameras[i], frame=int(frame))
                image = utils.downsample(
                        np.asarray(Image.open(imagepath), dtype=np.uint8), self.fixedcamdownsample).transpose((2, 0, 1)).astype(np.float32)
                fixedcamimage.append(image)
            fixedcamimage = np.concatenate(fixedcamimage, axis=1)
            fixedcamimage[:] -= self.fixedcammean
            fixedcamimage[:] /= self.fixedcamstd
            result["fixedcamimage"] = fixedcamimage

        # image from one or more cameras, always the same frame
        if "fixedframeimage" in self.keyfilter:
            ninput = len(self.fixedcameras)

            fixedframeimage = []
            for i in range(ninput):
                imagepath = self.imagepath.format(
                        seg=self.fixedframesegframe[0],
                        cam=self.fixedcameras[i],
                        frame=int(self.fixedframesegframe[1]))
                image = utils.downsample(
                        np.asarray(Image.open(imagepath), dtype=np.uint8), self.fixedcamdownsample).transpose((2, 0, 1)).astype(np.float32)
                fixedframeimage.append(image)
            fixedframeimage = np.concatenate(fixedframeimage, axis=1)
            fixedframeimage[:] -= self.fixedcammean
            fixedframeimage[:] /= self.fixedcamstd
            result["fixedframeimage"] = fixedframeimage

        # vertices
        for k in ["verts", "verts_next"]:
            if k in self.keyfilter:
                vertpath = self.vertpath.format(seg=seg, frame=int(frame) + (1 if k == "verts_next" else 0))
                verts = np.fromfile(vertpath, dtype=np.float32)
                if self.standardizeverts:
                    verts -= self.vertmean.ravel()
                    verts /= self.vertstd
                result[k] = verts.reshape((-1, 3))

        # texture averaged over all cameras for a single frame
        for k in ["avgtex", "avgtex_next"]:
            if k in self.keyfilter:
                texpath = self.texpath.format(seg=seg, cam="average", frame=int(frame) + (1 if k == "avgtex_next" else 0))
                try:
                    tex = np.asarray(Image.open(texpath), dtype=np.uint8)
                    if tex.shape[0] != self.avgtexsize:
                        tex = cv2.resize(tex, dsize=(self.avgtexsize, self.avgtexsize), interpolation=cv2.INTER_LINEAR)
                    tex = tex.transpose((2, 0, 1)).astype(np.float32)
                except:
                    tex = np.zeros((3, self.avgtexsize, self.avgtexsize), dtype=np.float32)
                    validinput = False
                if np.sum(tex) == 0:
                    validinput = False
                texmask = np.sum(tex, axis=0) != 0
                if self.standardizeavgtex:
                    tex -= self.avgtexmean
                    tex /= self.texstd
                    tex[:, ~texmask] = 0.
                result[k] = tex

        # keep track of whether we fail to load any of the input
        result["validinput"] = np.float32(1.0 if validinput else 0.0)

        if "modelmatrix" in self.keyfilter or "modelmatrixinv" in self.keyfilter or "camera" in self.keyfilter:
            def to4x4(m):
                return np.r_[m, np.array([[0., 0., 0., 1.]], dtype=np.float32)]

            # per-frame rigid transformation of scene/object
            for k in ["modelmatrix", "modelmatrix_next"]:
                if k in self.keyfilter:
                    if self.transfpath is not None:
                        transfpath = self.transfpath.format(seg=seg, frame=int(frame) + (1 if k == "modelmatrix_next" else 0))
                        try:
                            frametransf = np.genfromtxt(os.path.join(transfpath)).astype(np.float32)
                        except:
                            frametransf = None

                        result[k] = to4x4(np.dot(
                            np.linalg.inv(to4x4(frametransf)),
                            to4x4(self.basetransf))[:3, :4])
                    else:
                        result[k] = np.eye(3, 4, dtype=np.float32)#np.linalg.inv(to4x4(self.basetransf))[:3, :4]

            # inverse of per-frame rigid transformation of scene/object
            for k in ["modelmatrixinv", "modelmatrixinv_next"]:
                if k in self.keyfilter:
                    if self.transfpath is not None:
                        transfpath = self.transfpath.format(seg=seg, frame=int(frame) + (1 if k == "modelmatrixinv_next" else 0))
                        try:
                            frametransf = np.genfromtxt(os.path.join(transfpath)).astype(np.float32)
                        except:
                            frametransf = None

                        result[k] = to4x4(np.dot(
                            np.linalg.inv(to4x4(self.basetransf)),
                            to4x4(frametransf))[:3, :4])
                    else:
                        result[k] = np.eye(3, 4, dtype=np.float32)#self.basetransf

        # camera-specific data
        if cam is not None:
            # camera pose
            if "camera" in self.keyfilter:
                result["campos"] = np.dot(self.basetransf[:3, :3].T, self.campos[cam] - self.basetransf[:3, 3])
                result["camrot"] = np.dot(self.basetransf[:3, :3].T, self.camrot[cam].T).T
                result["focal"] = self.focal[cam]
                result["princpt"] = self.princpt[cam]
                result["camindex"] = self.allcameras.index(cam)

            # per-frame / per-camera unwrapped texture map
            if "tex" in self.keyfilter:
                texpath = self.texpath.format(seg=seg, cam=cam, frame=frame)
                try:
                    tex = np.asarray(Image.open(texpath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
                except:
                    tex = np.zeros((3, self.texsize, self.texsize), dtype=np.float32)

                assert tex.shape[1] == self.texsize
                texmask = np.sum(tex, axis=0) != 0
                if self.standardizetex:
                    tex -= self.texmean
                    tex /= self.texstd
                    tex[:, ~texmask] = 0.
                result["tex"] = tex
                result["texmask"] = texmask

            # camera images
            if "image" in self.keyfilter:
                # target image
                imagepath = self.imagepath.format(seg=seg, cam=cam, frame=int(frame))
                image = utils.downsample(
                        np.asarray(Image.open(imagepath), dtype=np.uint8),
                        self.downsample).transpose((2, 0, 1)).astype(np.float32)
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)

                # remove black level
                result["image"] = np.clip(image - np.array(self.blacklevel, dtype=np.float32)[:, None, None], 0., None)
                result["imagevalid"] = valid

                # optionally mask pixels with bright background values
                if self.maskbrightbg and cam in self.bg:
                    result["imagemask"] = self.bg[cam][1]

                # optionally mask pixels with bright values
                if self.maskbright:
                    if "imagemask" in result:
                        result["imagemask"] *= np.where(
                                (image[0] > 245.) |
                                (image[1] > 245.) |
                                (image[2] > 245.), 0., 1.)[None, :, :]
                    else:
                        result["imagemask"] = np.where(
                                (image[0] > 245.) |
                                (image[1] > 245.) |
                                (image[2] > 245.), 0., 1.).astype(np.float32)[None, :, :]

            # background image
            if "bg" in self.keyfilter and self.returnbg:
                result["bg"] = self.bg[cam][0]

            # image pixel coordinates
            if "pixelcoords" in self.keyfilter:
                if self.subsampletype == "patch":
                    indx = torch.randint(0, width - self.subsamplesize + 1, size=(1,)).item()
                    indy = torch.randint(0, height - self.subsamplesize + 1, size=(1,)).item()

                    py, px = torch.meshgrid(
                            torch.arange(indy, indy + self.subsamplesize).float(),
                            torch.arange(indx, indx + self.subsamplesize).float())
                elif self.subsampletype == "random":
                    px = torch.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).float()
                    py = torch.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).float()
                elif self.subsampletype == "random2":
                    px = torch.random(size=(self.subsamplesize, self.subsamplesize)).float() * (width - 1)
                    py = torch.random(size=(self.subsamplesize, self.subsamplesize)).float() * (height - 1)
                elif self.subsampletype == "stratified":
                    ssy = self.subsamplesize
                    ssx = self.subsamplesize
                    bsizex = (width - 1.) / ssx
                    bsizey = (height - 1.) / ssy
                    px = (torch.arange(ssx)[None, :] + torch.rand(size=(ssy, ssx))) * bsizex
                    py = (torch.arange(ssy)[:, None] + torch.rand(size=(ssy, ssx))) * bsizey
                elif self.subsampletype == None:
                    py, px = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float())
                else:
                    raise

                result["pixelcoords"] = torch.stack([px, py], dim=-1)

        return result
