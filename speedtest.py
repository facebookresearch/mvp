# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Measure time to evaluate network."""
import argparse
from contextlib import nullcontext
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
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
    parser.add_argument('--nofuse', action='store_true', help='don\'t call apply(fuse)')
    parser.add_argument('--scripting', action='store_true', help='use torch.jit.script')
    parser.add_argument('--profiler', type=str, help='use PyTorch profiler, write trace to filename')
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
            batch_size=args.batchsize, shuffle=False, num_workers=16)

    # build autoencoder
    ae = profile.get_autoencoder(dataset)
    ae = ae.to("cuda").eval()

    # load
    state_dict = ae.state_dict()
    trained_state_dict = torch.load("{}/aeparams.pt".format(outpath))
    trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
    state_dict.update(trained_state_dict)
    ae.load_state_dict(state_dict, strict=False)

    if not args.nofuse:
        ae.apply(fuse(trainiter, profile.get_ae_args()["renderoptions"]))
    ae.apply(no_grad)

    if args.scripting:
        ae.encoder = torch.jit.script(ae.encoder)
        ae.decoder = torch.jit.script(ae.decoder)

    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()

    enctime = []
    dectime = []
    rmtime = []
    bgtime = []
    totaltime = []

    torch.cuda.synchronize()

    if args.profiler is not None:
        cm = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True, profile_memory=False)
    else:
        cm = nullcontext()

    with cm as prof:
        with torch.no_grad():
            for data in dataloader:
                b = utils.findbatchsize(data)

                # forward
                datacuda = utils.tocuda(data)

                torch.cuda.synchronize()
                t0 = time.time()

                output, _ = ae(
                        trainiter=trainiter,
                        evaliter=itemnum +  torch.arange(b, device="cuda"),
                        outputlist=["enctime", "dectime", "rmtime", "bgtime"],
                        losslist=[],
                        **datacuda,
                        **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

                torch.cuda.synchronize()
                t1 = time.time()

                enctime.append(output["enctime"])
                dectime.append(output["dectime"])
                rmtime.append(output["rmtime"])
                if "bgtime" in output:
                    bgtime.append(output["bgtime"])
                totaltime.append(t1 - t0)

                endtime = time.time()
                ips = 1. / (endtime - starttime)
                print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
                starttime = time.time()

                iternum += 1
                itemnum += b

    enctime = np.array(enctime)
    dectime = np.array(dectime)
    rmtime = np.array(rmtime)
    bgtime = np.array(bgtime)
    totaltime = np.array(totaltime)

    print("encode", 1000. * np.median(enctime[10:]))
    print("decode", 1000. * np.median(dectime[10:]))
    print("raymarch", 1000. * np.median(rmtime[10:]))
    if len(bgtime) > 0:
        print("bg", 1000. * np.median(bgtime[10:]))
    print("totaltime", 1000. * np.median(totaltime[10:]))
    difftime = totaltime - enctime - dectime - rmtime - (bgtime if len(bgtime) > 0 else 0.)
    print("difftime", 1000. * np.median(difftime[10:]))

    if args.profiler is not None:
        prof.export_chrome_trace(args.profiler)
