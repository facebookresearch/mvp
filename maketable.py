# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cmap
import matplotlib.pyplot as plt

paths = [
("mvp",          "r", "r",  "o",  48,  1.,  "experiments/example2/mvp"),
]

plt.rc('font', size=8)
plt.figure(figsize=(6.4*2, 4.8*2))
ax = plt.gca()

xaxis = (2, "PSNR")

with open("table.csv", "w") as f:
    for i, (expname, facecolor, edgecolor, marker, size, alpha, path) in enumerate(paths):
        camrow = []
        cols = []
        vals = []
        for cam in ["400291"]:
            cols_, vals_ = np.genfromtxt(os.path.join(path, "mse_{}.txt".format(cam)),
                    dtype=None, encoding='utf-8', unpack=True)
            camrow.append(cam)
            camrow.extend([""]*(len(cols_)-1))
            cols.extend(cols_)
            vals.extend(vals_)
            camrow.append("")
            cols.append("")
            vals.append("")

            ax.plot([vals_[xaxis[0]], vals_[xaxis[0]]], [0., vals_[4]], '-y', linewidth=0.5, zorder=1)
            ax.plot([vals_[xaxis[0]], vals_[xaxis[0]]], [vals_[4], vals_[4]+vals_[5]], '-m', linewidth=0.5, zorder=1)
            ax.scatter(vals_[xaxis[0]], vals_[4] + vals_[5],
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    marker=marker,
                    s=size,
                    alpha=1.,
                    label="{} {:.2f}".format(expname, vals_[0] / 1000000.),
                    zorder=2)

        if i == 0:
            print("," + ",".join([str(x) for x in camrow]), file=f)
            print("," + ",".join([str(x) for x in camrow]))
            print("," + ",".join([str(x) for x in cols]), file=f)
            print("," + ",".join([str(x) for x in cols]))
        print(expname + "," + ",".join([str(x) for x in vals]), file=f)
        print(expname + "," + ",".join([str(x) for x in vals]))

plt.legend()
plt.title("perf")
plt.xlabel(xaxis[1])
plt.ylabel("decode + raymarch time (ms)")
plt.xlim(plt.xlim()[0], plt.xlim()[1]+0.001)
plt.ylim(0., plt.ylim()[1]+1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.savefig("tablefig_{}.png".format(cam))
