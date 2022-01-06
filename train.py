# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Train an autoencoder."""
import argparse
import importlib
import importlib.util
import os
import sys
import time
sys.dont_write_bytecode = True

import numpy as np

import torch
import torch.utils.data
torch.backends.cudnn.benchmark = True # gotta go fast!

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
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    if args.resume:
        iternum = utils.findmaxiters("{}/log.txt".format(outpath))
    else:
        iternum = 0
    log = utils.Logger("{}/log.txt".format(outpath), "a" if args.resume else "w")
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    starttime = time.time()
    experconfig = utils.import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    if not args.noprogress:
        progressprof = experconfig.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()
    if not args.noprogress:
        testdataset = progressprof.get_dataset()
        dataloader = torch.utils.data.DataLoader(testdataset,
                batch_size=progressprof.batchsize, shuffle=False,
                drop_last=True, num_workers=0)
        for testbatch in dataloader:
            break

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
    if not args.noprogress:
        writer = progressprof.get_writer()
        print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = ae.to("cuda").train()
    if args.resume:
        ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    # compile with jit
    if args.scripting:
        ae.encoder = torch.jit.script(ae.encoder)
        ae.decoder = torch.jit.script(ae.decoder)

    # build optimizer
    starttime = time.time()
    optim = profile.get_optimizer(ae)
    if args.resume:
        optim.load_state_dict(torch.load("{}/optimparams.pt".format(outpath)))
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    prevloss = np.inf

    for epoch in range(10000):
        for data in dataloader:
            # forward
            cudadata = utils.tocuda(data)
            output, losses = ae(
                    trainiter=iternum,
                    outputlist=profile.get_outputlist(),
                    losslist=lossweights.keys(),
                    **cudadata,
                    **(profile.get_ae_args() if hasattr(profile, "get_ae_args") else {}))

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in losses.items()])

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

            # update parameters
            optim.zero_grad()
            loss.backward()
            optim.step()

            # compute evaluation output
            if not args.noprogress and iternum in evalpoints:
                with torch.no_grad():
                    testoutput, _ = ae(
                            trainiter=iternum,
                            outputlist=progressprof.get_outputlist() + ["rmtime"],
                            losslist=[],
                            **utils.tocuda(testbatch),
                            **progressprof.get_ae_args())

                print("Iteration {}: rmtime = {:.5f}".format(iternum, testoutput["rmtime"] * 1000.))

                writer.batch(iternum, iternum * profile.batchsize + torch.arange(0), **testbatch, **testoutput)

            if not args.nostab and (loss.item() > 400 * prevloss or not np.isfinite(loss.item())):
                print("unstable loss function; resetting")

                ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                optim = profile.get_optimizer(ae)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 1000 == 0:
                torch.save(ae.state_dict(), "{}/aeparams.pt".format(outpath))
                torch.save(optim.state_dict(), "{}/optimparams.pt".format(outpath))

            if iternum >= profile.maxiter:
                break

            iternum += 1

        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
