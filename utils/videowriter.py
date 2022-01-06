# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Write images to a tmp folder and then use ffmpeg to create a video"""
import os
import shutil
import subprocess

import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import cv2
cv2.setNumThreads(0)

import torch
import torch.nn.functional as F

def sinebow(h):
    h += 1/2
    h *= -1
    r = np.sin(np.pi * h) * 0.75
    g = np.sin(np.pi * (h + 1/3)) * 0.65
    b = np.sin(np.pi * (h + 2/3))
    return [int(255*chan**2) for chan in (r, g, b)]

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255

    return np.array([sinebow(x) for x in np.linspace(0., 1., ncols, endpoint=False)])

def flow_compute_color(flow_uv, convert_to_bgr=False):
    u = flow_uv[0]
    v = flow_uv[1]

    flow_image = torch.zeros(3, u.size(0), u.size(1), device=flow_uv.device)

    colorwheel = torch.tensor(make_colorwheel()).to(device=flow_uv.device)
    ncols = colorwheel.size(1)

    rad = torch.sqrt(torch.square(u) + torch.square(v)) * 0.5
    a = (torch.atan2(-v, -u)/torch.pi).clamp(min=-1., max=1.)

    fk = (a+1) / 2*(ncols-1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1.data[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.size(1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        col = (1. - rad * (1. - col)).clamp(min=0.001, max=0.999)

        flow_image[i,:,:] = 255. * col

    return flow_image

def resizecat(arr, dim=0):
    height, width = arr[0].size(1), arr[0].size(2)

    for i in range(1, len(arr)):
        if arr[i].size(1) != height:
            arr[i] = F.interpolate(
                    arr[i][None],#.permute(0, 3, 1, 2),
                    size=(height, int(arr[i].size(2) * height / arr[i].size(1))),
                    mode='bilinear',
                    align_corners=False)[0]#.permute(0, 2, 3, 1)

    return torch.cat(arr, dim=dim)

def writeimage(randid, itemnum, outputs, **kwargs):
    imgout = []

    for k in outputs:
        data = outputs[k]

        if isinstance(data, str):
            img = imgout[-1].permute(1, 2, 0).data.numpy().copy('C')
            cv2.putText(img, data, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            imgout[-1] = torch.from_numpy(img).permute(2, 0, 1)
        elif len(data.size()) == 3 and data.size(0) == 3: # rgb image
            data = data * torch.tensor(kwargs["colcorrect"][:, None, None])
            imgout.append(data)
        elif len(data.size()) == 3 and data.size(0) == 1: # 1 channel image
            cmap = kwargs["cmap"]
            cmapscale = kwargs["cmapscale"]
            colorbar = kwargs["colorbar"]
            if cmap is None:
                imgout.append(data.repeat(3, 1, 1))
            else:
                cmap = cm.get_cmap(cmap)

                colors = cmap.colors if isinstance(cmap, matplotlib.colors.ListedColormap) else cmap(range(256))
                colors = torch.tensor(colors, device=data.device).permute(1, 0)
                indices = (data[0] * 255. / cmapscale).clamp(min=0., max=255.).long()
                data = colors[:, indices][:3, :, :] * 255.

                if colorbar is not None:
                    data = torch.cat([
                        data,
                        torch.zeros(3, data.size(1), colorbar.size(2), device=data.device)
                        ], dim=2)
                    cbarstarty = data.size(1)//2 - colorbar.size(1)//2
                    data.data[:, cbarstarty:cbarstarty+colorbar.size(1), -colorbar.size(2):] = \
                            colorbar[:3, :, :]

                imgout.append(data)
        elif len(data.size()) == 3 and data.size(0) == 2: # flow field?
            yvals, xvals = torch.meshgrid(
                    torch.linspace(-1., 1., 128, device=data.device),
                    torch.linspace(-1., 1., 128, device=data.device))
            vals = torch.stack([xvals, yvals], axis=-1)

            data[:, :128, :128].data = vals
            imgout.append(flow_compute_color(data))

    imgout = resizecat(imgout, dim=2)

    if imgout.size(2) % 2 != 0:
        imgout = imgout[:, :, :-1]
    if imgout.size(1) % 2 != 0:
        imgout = imgout[:, :-1, :]

    # apply gamma
    imgout = ((imgout / 255.) ** (1. / 1.8) * 255.).clamp(min=0., max=255.).byte()

    # CHW -> HWC
    imgout = imgout.permute(1, 2, 0)

    imgout = imgout.data.to("cpu").numpy()

    Image.fromarray(imgout).save("{}/{:06}.png".format(randid, itemnum))

class Writer():
    def __init__(self, outpath, keyfilter,
            bgcolor=[0., 0., 0.],
            #colcorrect=[1.35, 1.16, 1.5],
            colcorrect=[1., 1., 1.],
            cmap=None,
            cmapscale=255.,
            colorbar=False,
            nthreads=16):
        self.outpath = outpath
        self.keyfilter = keyfilter
        self.bgcolor = np.array(bgcolor, dtype=np.float32)
        self.colcorrect = np.array(colcorrect, dtype=np.float32)
        self.cmap = cmap
        self.cmapscale = cmapscale
        self.colorbar = colorbar

        if cmap is not None and colorbar:
            plt.style.use('dark_background')
            a = np.array([[0,self.cmapscale]])
            fig = plt.figure(figsize=(1, 9))
            img = plt.imshow(a, cmap=cmap)
            plt.gca().set_visible(False)
            cax = plt.axes([0.1, 0.2, 0.2, 0.8])
            plt.colorbar(orientation="vertical", cax=cax)
            plt.savefig("/tmp/colorbar.png", bbox_inches='tight')
            self.colorbarimg = np.asarray(Image.open("/tmp/colorbar.png"), dtype=np.float32)
            self.colorbarimg = torch.from_numpy(self.colorbarimg).permute(2, 0, 1)
        else:
            self.colorbarimg = None

        # set up temporary output
        self.randid = ''.join([str(x) for x in np.random.randint(0, 9, size=10)])
        try:
            os.makedirs("/tmp/{}".format(self.randid))
        except OSError:
            pass

        self.writepool = torch.multiprocessing.Pool(nthreads)
        self.nitems = 0

        self.encodings = []
        self.headpose = []
        self.lightpow = []

        self.asyncresults = []

    def batch(self, iternum, itemnum, **kwargs):
        b = itemnum.size(0)

        if "encoding" in self.keyfilter:
            self.encodings.extend(kwargs["encoding"].data.to("cpu").numpy().tolist())

        if "headpose" in self.keyfilter:
            self.headpose.extend(kwargs["headpose"].data.to("cpu").numpy().reshape((
                kwargs["headpose"].size(0), -1)).tolist())

        if "lightpow" in self.keyfilter:
            self.lightpow.extend(kwargs["lightpow"].data.to("cpu").numpy().tolist())

        for i in range(b):
            self.asyncresults.append(
                self.writepool.apply_async(writeimage,
                    ("/tmp/{}".format(self.randid), itemnum[i], {k:
                        kwargs[k][i].data.to("cpu") if isinstance(kwargs[k][i], torch.Tensor) else kwargs[k][i]
                        for k in self.keyfilter}),
                    {"cmap": self.cmap, "cmapscale": self.cmapscale,
                        "colorbar": self.colorbarimg,
                        "colcorrect": self.colcorrect}))
        self.nitems += b

    def finalize(self):
        if len(self.encodings) > 0:
            np.savetxt(self.outpath[:-4] + "_enc.txt", self.encodings)
        if len(self.headpose) > 0:
            np.savetxt(self.outpath[:-4] + "_headpose.txt", self.headpose)
        if len(self.lightpow) > 0:
            np.savetxt(self.outpath[:-4] + "_lightpow.txt", self.lightpow)

        for r in self.asyncresults:
            if r is not None:
                r.wait()

        if self.nitems == 1:
            os.system("cp /tmp/{}/{:06}.png {}.png".format(self.randid, 0, self.outpath[:-4]))
        elif self.nitems > 0:
            # make video file
            command = (
                    "ffmpeg -y -r 30 -i /tmp/{}/%06d.png "
                    "-vframes {} "
                    "-vcodec libx264 -crf 18 "
                    "-pix_fmt yuv420p "
                    "{}".format(self.randid, self.nitems, self.outpath)
                    ).split()
            subprocess.call(command)

            shutil.rmtree("/tmp/{}".format(self.randid))
