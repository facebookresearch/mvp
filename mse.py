# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate MSE and runtime."""
import argparse
import importlib
import importlib.util
import os
import re
import sys
import time
sys.dont_write_bytecode = True

import numpy as np

import torch
import torch.utils.data
torch.backends.cudnn.benchmark = True # gotta go fast!
import torch.nn.functional as F

import torch.autograd.profiler as profiler

from utils import utils
from models.utils import fuse, no_grad

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config')
    parser.add_argument('--profile', type=str, default='Eval', help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--batchsize', type=int, default=1, help='batchsize')
    parser.add_argument('--scripting', action='store_true', help='use torch.jit.script')
    parser.add_argument('--outfilesuffix', type=str, default='', help='output file name suffix')
    parser.add_argument('--usemask', action='store_true', help='use imagemask from dataset')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load log
    trainiter = utils.findmaxiters(os.path.join(outpath, "log.txt"))

    # load config
    experconfig = utils.import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})

    # load datasets
    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batchsize, shuffle=False,
            num_workers=8 if not hasattr(args, 'maxframes') or args.maxframes != 1 else 0)

    # build autoencoder
    ae = profile.get_autoencoder(dataset)
    ae = ae.to("cuda").eval()

    # load
    state_dict = ae.state_dict()
    trained_state_dict = torch.load("{}/aeparams.pt".format(outpath))
    trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
    state_dict.update(trained_state_dict)
    ae.load_state_dict(state_dict, strict=False)

    ae.apply(fuse(trainiter, profile.get_ae_args()["renderoptions"]))
    ae.apply(no_grad)

    if args.scripting:
        ae.encoder = torch.jit.script(ae.encoder)
        ae.decoder = torch.jit.script(ae.decoder)

    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()

    mse = 0.
    nsamples = 0.
    ssim = 0.

    dectime = []
    rmtime = []

    print("training iters:", trainiter)
    print()

    with torch.no_grad():
        for data in dataloader:
            b = utils.findbatchsize(data)

            # forward
            datacuda = utils.tocuda(data)
            output, _ = ae(
                    trainiter=trainiter,
                    evaliter=itemnum +  torch.arange(b, device="cuda"),
                    outputlist=(profile.get_outputlist() if hasattr(profile, "get_outputlist") else []) +
                        ["dectime", "rmtime", "image", "irgbrec"],
                    losslist=[],
                    **datacuda,
                    **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

            dectime.append(output["dectime"])
            rmtime.append(output["rmtime"])

            irgbrec = output["irgbrec"]
            image = datacuda["image"]

            if args.usemask:
                # compute reconstruction loss weighting
                weight = torch.ones_like(image) * datacuda["validinput"][:, None, None, None]
                if "imagevalid" in datacuda and datacuda["imagevalid"] is not None:
                    weight = weight * datacuda["imagevalid"][:, None, None, None]
                if "imagemask" in datacuda and datacuda["imagemask"] is not None:
                    weight = weight * datacuda["imagemask"]

                mse += float(torch.sum((weight * (irgbrec - image) ** 2)).item())
                nsamples += float(torch.sum(weight).item())
            else:
                mse += float(torch.sum((irgbrec - image) ** 2).item())
                nsamples += float(image.numel())

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\r")
            starttime = time.time()

            iternum += 1
            itemnum += b

    print()

    mse = mse / nsamples
    psnr = 20. * np.log10(255.) - 10. * np.log10(mse)

    with open(os.path.join(outpath, "mse" + args.outfilesuffix + ".txt"), "w") as f:
        for outdest in [f, sys.stdout]:
            print("iter", trainiter, file=outdest)
            print("MSE", mse, file=outdest)
            print("PSNR", psnr, file=outdest)

            print("decode", 1000. * np.median(dectime[5:]), file=outdest)
            print("raymarch", 1000. * np.median(rmtime[5:]), file=outdest)
