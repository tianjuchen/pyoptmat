#!/usr/bin/env python3

import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def read_file(path, ltype="time", opt="adam"):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        ref = os.path.basename(f).split("-")[0]
        optimizer = os.path.basename(f).split("-")[1].split(".txt")[0]
        if ref == ltype and optimizer == opt:
            df = pd.read_csv(f)
            return df


if __name__ == "__main__":

    # Load moose calculated results
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/"
    path2 = "examples/structural-inference/tension/time/batch-15-5/"
    path = path1 + path2

    batch1_time = read_file(path, ltype="time", opt="adam")
    batch1_loss = read_file(path, ltype="loss", opt="adam")

    batch2_time = read_file(path, ltype="time", opt="newton")
    batch2_loss = read_file(path, ltype="loss", opt="newton")


    # Plot results against each case
    ax = plt.gca()
    plt.plot(batch1_time[1:], batch1_loss, lw=3, label="Adam")
    plt.plot(batch2_time[1:], batch2_loss, lw=7, alpha=0.75, label="LBFGS")
    # plt.yscale("log")
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xlim(0, 1000)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.locator_params(axis='both', nbins=4)
    # plt.legend(loc="best", frameon=False, prop={"size": 18})
    plt.legend(
        [
            Line2D([0], [0], lw=5, color='#1f77b4'),
            Line2D([0], [0], lw=5, color='#ff7f0e'),
        ],
        ["Adam", "L-BFGS"],
        loc="best", prop={'size': 16},
        frameon=False,
    )
    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    plt.savefig(path + "time-batch-15-5.pdf")
    plt.show()
    plt.close()
