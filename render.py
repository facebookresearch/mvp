# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Render object from training camera viewpoint or novel viewpoints."""
import argparse
from contextlib import nullcontext
import importlib
import importlib.util
import os
import re
import sys
import time
sys.dont_write_bytecode = True

import torch
torch.backends.cudnn.benchmark = True # gotta go fast!
import torch.nn.functional as F
import torch.utils.data
import torch.profiler

from utils import utils
from models.utils import fuse, no_grad

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # parse arguments
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config')
    parser.add_argument('--profile', type=str, default='Eval', help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
    parser.add_argument('--nofuse', action='store_true', help='don\'t call apply(fuse)')
    parser.add_argument('--scripting', action='store_true', help='use torch.jit.script')
    parser.add_argument('--profiler', type=str, help='use pytorch profiler, write trace to filename')
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
    if hasattr(profile, "get_dataset_sampler"):
        dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batchsize,
                sampler=profile.get_dataset_sampler(dataset), drop_last=False,
                num_workers=8)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batchsize, shuffle=False, drop_last=False,
                num_workers=8 if not hasattr(args, 'maxframes') or args.maxframes != 1 else 0)

    # data writer
    writer = profile.get_writer()

    # build autoencoder
    ae = profile.get_autoencoder(dataset)
    ae = ae.to("cuda").eval()

    # load
    state_dict = ae.state_dict()
    trained_state_dict = torch.load("{}/aeparams.pt".format(outpath))
    trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
    state_dict.update(trained_state_dict)
    ae.load_state_dict(state_dict, strict=False)

    # compute total number of params
    total = 0.
    for k, v in trained_state_dict.items():
        print(k, v.numel() * 4. / 1e9)
        total += v.numel() * 4. / 1e9
    print(total, "B params")
    print("loaded params")

    if not args.nofuse:
        ae.apply(fuse(trainiter, profile.get_ae_args()["renderoptions"]))
    ae.apply(no_grad)
    print("fuse done")

    if args.scripting:
        ae.encoder = torch.jit.script(ae.encoder)
        ae.decoder = torch.jit.script(ae.decoder)
        print("torch.jit.script done")

    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()

    if args.profiler:
        cm = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True, profile_memory=True)
    else:
        cm = nullcontext()

    with cm as prof:
        with torch.inference_mode():
            for data in dataloader:
                b = utils.findbatchsize(data)

                # forward
                datacuda = utils.tocuda(data)
                output, _ = ae(
                        trainiter=trainiter,
                        evaliter=itemnum +  torch.arange(b, device="cuda"),
                        outputlist=profile.get_outputlist() if hasattr(profile, "get_outputlist") else [],
                        losslist=[],
                        **datacuda,
                        **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

                writer.batch(iternum, itemnum + torch.arange(b), **datacuda, **output)

                endtime = time.time()
                ips = 1. / (endtime - starttime)
                print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
                starttime = time.time()

                iternum += 1
                itemnum += b

        # cleanup
        writer.finalize()

    if args.profiler is not None:
        prof.export_chrome_trace(args.profiler)
