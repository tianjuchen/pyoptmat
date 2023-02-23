#!/usr/bin/env python3

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path
import xarray as xr
import torch
from maker import make_model, load_subset_data
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pyoptmat import optimize, experiments
# Don't care if integration fails
import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


def read_file(path, file_name):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        fn = os.path.basename(f).split(".txt")[0]
        if fn == file_name:
            df = pd.read_csv(f)
            return df



def time_back():
    nbacks = np.array([1, 2, 3])
    time_adam = np.array([2780, 3.16e3, 4439])
    time_lbfgs = np.array([3060, 3.92e3, 5977])
    time_2point = np.array([4200, 1.6e4, 6.68e4])
    
    plt.plot(nbacks, time_adam, '-o', lw=4, markersize=20, label="Adam")
    plt.plot(nbacks, time_lbfgs, '-o', lw=4, markersize=20, label="L-BFGS")
    plt.plot(nbacks, time_2point, '-o', lw=4, markersize=20, label="L-BFGS (2 point)")
    # plt.yscale("log")
    
    ax = plt.gca()
    plt.xlabel(r"$n_{back}$", fontsize=23)
    plt.ylabel("Time", fontsize=23)
    plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    # plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.tick_params(axis="both", which="major", labelsize=23)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("time-nback.pdf")
    plt.show()
    plt.close()
    return nbacks


def plot_score():
    df = pd.DataFrame(
        {
            "accuracy": [91.96, 98.06, 98.45, 98.70, 99.88, 100.0, 99.38, 99.88, 99.96],
            "optimizer": ["Adam"] * 3 + ["L-BFGS"] * 3 + ["L-BFGS (2 point)"] * 3,
            "nback": ["1", "2", "3"] * 3
        }
    )
    print(df)
    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="nback", y="accuracy", hue="optimizer")
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 16})
    ax = plt.gca()
    plt.ylim(0, 124)
    plt.xlabel(r"$n_{back}$", fontsize=23)
    plt.ylabel("Strict Accuracy Score (%)", fontsize=23)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("score-nback.pdf")
    plt.show()
    plt.close()
    return df


if __name__ == "__main__":

    _ = plot_score()
    # _ = time_back()