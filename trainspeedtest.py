# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate speed of training loop."""
import argparse
from contextlib import nullcontext
import gc
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

import torch.autograd.profiler as profiler

from utils import utils

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default='Train', help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--noprogress', action='store_true', help='don\'t output training progress images')
    parser.add_argument('--nostab', action='store_true', help='don\'t check loss stability')
    parser.add_argument('--scripting', action='store_true', help='use torch.jit.script')
    parser.add_argument('--profiler', type=str, help='use PyTorch profiler, write trace to filename')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    iternum = utils.findmaxiters("{}/log.txt".format(outpath))
    print("iternum={}".format(iternum))

    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    starttime = time.time()
    experconfig = utils.import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    print("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()
    dataset = profile.get_dataset()
    print("len(dataset)=", len(dataset))
    if hasattr(profile, "get_dataset_sampler"):
        dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=profile.batchsize,
                sampler=profile.get_dataset_sampler(), drop_last=True,
                num_workers=8, persistent_workers=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=profile.batchsize, shuffle=True, drop_last=True,
                num_workers=8, persistent_workers=True)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = ae.to("cuda").train()
    ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    if args.scripting:
        ae.encoder = torch.jit.script(ae.encoder)
        ae.decoder = torch.jit.script(ae.decoder)

    # build optimizer
    starttime = time.time()
    optim = profile.get_optimizer(ae)
    #optim.load_state_dict(torch.load("{}/optimparams.pt".format(outpath)))
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    prevloss = np.inf

    tocudat = []
    fwdt = []
    enct = []
    dect = []
    rmt = []
    bgt = []
    losst = []
    bwdt = []
    stept = []
    totalt = []

    niter = 0

    torch.cuda.synchronize()

    if args.profiler is not None:
        cm = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True, profile_memory=False)
    else:
        cm = nullcontext()

    with cm as prof:
        for data in dataloader:
            torch.cuda.synchronize()
            t0 = time.time()

            # forward
            cudadata = utils.tocuda(data)

            torch.cuda.synchronize()
            t1 = time.time()

            output, losses = ae(
                    trainiter=iternum,
                    outputlist=profile.get_outputlist() + ["enctime", "dectime", "rmtime", "bgtime"],
                    losslist=lossweights.keys(),
                    **cudadata,
                    **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

            torch.cuda.synchronize()
            t2 = time.time()

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in losses.items()])

            torch.cuda.synchronize()
            t3 = time.time()

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                    ", ".join(["{} = {:.5f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in losses.items()]), end="")
            if iternum % 10 == 0:
                endtime = time.time()
                ips = 10. / (endtime - starttime)
                print(", iter/sec = {:.2f}".format(ips))
                starttime = time.time()
            else:
                print()

            torch.cuda.synchronize()
            t4 = time.time()

            # update parameters
            optim.zero_grad()
            loss.backward()

            torch.cuda.synchronize()
            t5 = time.time()

            optim.step()

            torch.cuda.synchronize()
            t6 = time.time()

            prevloss = loss.item()

            if iternum >= profile.maxiter:
                break

            iternum += 1

            torch.cuda.synchronize()
            t7 = time.time()

            tocudat.append((t1 - t0) * 1000.)
            fwdt.append((t2 - t1) * 1000.)
            enct.append(output["enctime"] * 1000.)
            dect.append(output["dectime"] * 1000.)
            rmt.append(output["rmtime"] * 1000.)
            if "bgtime" in output:
                bgt.append(output["bgtime"] * 1000.)
            losst.append((t3 - t2) * 1000.)
            bwdt.append((t5 - t4) * 1000.)
            stept.append((t6 - t5) * 1000.)
            totalt.append((t7 - t0) * 1000.)

            niter += 1

            if niter % 30 == 0:
                print(niter)
                print("tocuda", np.median(np.array(tocudat)[20:]))
                print("fwd", np.median(np.array(fwdt)[20:]))
                print("  enc", np.median(np.array(enct)[20:]))
                print("  dec", np.median(np.array(dect)[20:]))
                print("  rm", np.median(np.array(rmt)[20:]))
                print("  bg", np.median(np.array(bgt)[20:]))
                print("loss", np.median(np.array(losst)[20:]))
                print("bwd", np.median(np.array(bwdt)[20:]))
                print("step", np.median(np.array(stept)[20:]))
                print("total", np.median(np.array(totalt)[20:]))

                if args.profiler is not None:
                    break

    if args.profiler is not None:
        prof.export_chrome_trace(args.profiler)
