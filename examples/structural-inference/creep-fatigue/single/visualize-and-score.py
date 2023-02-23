#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path

import numpy.random as ra
import numpy as np
import xarray as xr
import torch
import pandas as pd
import seaborn as sns
from maker import make_model, load_subset_data

from pyoptmat import optimize, experiments
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import time

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

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


def Accuracy(pred, res, sf=0.05):
    tmax = torch.amax(res, dim=0)
    pmax = torch.amax(pred, dim=0)
    pred = torch.abs(pred) / torch.abs(pmax)
    res = torch.abs(res) / torch.abs(tmax)
    correct = 0
    for i in range(res.shape[-1]):
        # correct += ((pred[:, i].int() == res[:, i].int()).sum().item()) / res.shape[0]
        cond = torch.logical_and(
            pred[:, i] <= res[:, i] * (1 + sf),
            pred[:, i] >= res[:, i] * (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def score_compare():

    df = pd.DataFrame(
        {
            "Score": np.array([99.01, 100.0]),
            "Time": np.array([1339.578, 2657.4285]),
            "Optimizers": ["Adam", "L-BFGS"],
        }
    )
    print(df)

    sns.set(font_scale=2.0, style="white")
    p = sns.barplot(
        y="Score",
        x="Optimizers",
        data=df,
        # capsize=0.4,
        # errcolor=".5",
        # linewidth=3,
        # edgecolor=".5",
        # facecolor=(0, 0, 0, 0),
    )
    sns.despine(top=False, right=False, left=False, bottom=False)
    ax = plt.gca()
    plt.ylim(0, 110)
    plt.locator_params(axis="x", nbins=4)
    plt.xlabel("", fontsize=23)
    plt.ylabel("", fontsize=23)
    
    # plt.legend(ncol=2, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    show_values(p, space=0)
    plt.tight_layout()
    plt.savefig("single-cyclic-score.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":

    score_compare()
    sys.exit("stop")
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.00
    nsamples = 1  # 20 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    print(data.shape)

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    """
    ics = [
        torch.tensor(0.5013),
        torch.tensor(0.5033),
        torch.tensor(0.5005),
        torch.tensor(0.5047),
        torch.tensor(0.5058),
        torch.tensor([0.4578, 0.4510, 0.3960]),
        torch.tensor([0.3823, 0.5575, 0.5578]),
    ]
    """

    ics = [
        torch.tensor(0.50),
        torch.tensor(0.50),
        torch.tensor(0.50),
        torch.tensor(0.50),
        torch.tensor(0.50),
        torch.tensor([0.50, 0.50, 0.50]),
        torch.tensor([0.50, 0.50, 0.50]),
    ]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    with torch.no_grad():
        pred = model(data, cycles, types, control)

    ax = plt.gca()
    plt.plot(data[2, :, -1].numpy(), pred[:, -1].numpy(), lw=3, label="L-BFGS")
    plt.plot(data[2, :, -1].numpy(), results[:, -1].numpy(), "k--", lw=3, label="True")

    plt.locator_params(axis="both", nbins=4)
    plt.xlabel("Strain", fontsize=18)
    plt.ylabel("Stress", fontsize=18)
    plt.legend(loc="best", ncol=1, frameon=False, prop={"size": 23})
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("single-cyclic-LBFGS-visualize.pdf")
    plt.show()
    plt.close()

    print("score is :", Accuracy(pred, results))
