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
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        ffname = os.path.basename(f).split(".csv")[0]
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


def convert_string_to_data(df):
    data = []
    for i in df:
        data.append(eval(i)[0])
    return np.array(data).mean()


if __name__ == "__main__":
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/statistical/hist/"

    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    rnames = [
        r"$n$",
        r"$\eta$",
        r"$s_{0}$",
        r"$R$",
        r"$d$",
        r"$C_{1}$",
        r"$C_{2}$",
        r"$C_{3}$",
        r"$g_{1}$",
        r"$g_{2}$",
        r"$g_{3}$",
    ]

    scale = "0.15"
    vns = ["mu", "std"]

    df = read_file(path1 + path2, "hist-" + scale + "-20")

    for i in range(len(rnames)):
        plt.style.use(latex_style_times)
        vn = names[i] + "_" + vns[1]
        plt.plot(df[vn][:300], "o-", lw=4, markersize=15, markevery=50, label=rnames[i])

    plt.plot(np.array([float(scale)] * len(df[vn][:300])), "k--", lw=4, label="True")
    # plt.xlim(-1, 300)
    # plt.ylim(0, 1)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=30)
    plt.ylabel(r"$\sigma^{2}$", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    # plt.legend(loc="upper right", frameon=False, ncol=3, prop={"size": 18})
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=3)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("cyclic-variance-hist-{}.pdf".format(float(scale)))
    plt.show()
    plt.close()
