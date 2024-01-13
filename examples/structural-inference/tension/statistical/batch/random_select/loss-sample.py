#!/usr/bin/env python3
import sys
import os, glob
import torch
import numpy as np
import xarray as xr
import os.path
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import scipy.signal as sig
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import pyro.distributions as dist
from matplotlib import RcParams


def read_file(path, file_name):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        ffname = os.path.basename(f).split(".txt")[0]
        if ffname == file_name:
            df = pd.read_csv(f)
            return df


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)


def calavg(mus):
    avg_mu = []
    for mu in mus:
        v = np.mean([df[mu].to_numpy() for df in dfs])
        avg_mu.append(v)
    return avg_mu


if __name__ == "__main__":
    path1 = "/mnt/c/Users/chent/Desktop/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/batch/random_select/"
    fpath = path1 + path2

    fnames = [
        "loss-history-amount-5",
        "loss-history-amount-10",
        "loss-history-amount-20",
        "loss-history-amount-30",
    ]

    sample_size = [5, 10, 20, 30]

    for fn, sz in zip(fnames, sample_size):
        df = read_file(fpath, fn)
        # plt.style.use(latex_style_times)
        plt.plot(df / sz, lw=3, label=r"$n_{sample}$" + r"$={}$".format(sz))
    # plt.yscale("log")
    fsize = 23
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel(r"$ELBO$", fontsize=fsize)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    ax.locator_params(nbins=4, axis="x")
    plt.legend(frameon=False, prop={"size": 25})
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig(fpath + "loss-sample.pdf")
    plt.show()
    plt.close()
