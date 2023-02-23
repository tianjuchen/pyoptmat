#!/usr/bin/env python3

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path
import xarray as xr
import torch
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


def read_file(path, fn):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        file = os.path.basename(f).split(".txt")[0]
        if file == fn:
            df = pd.read_csv(f)
            return df


def plot_score():
    df = pd.DataFrame(
        {
            "magnitude": np.array([2.06, 2.05, 2.44]),
            "varname": np.array([10, 20, 30]),
        }
    )
    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="varname", y="magnitude")
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 20})
    ax = plt.gca()
    plt.xlabel(r"$n_{sample}$", fontsize=18)
    plt.ylabel("Strict Accuracy Score (%)", fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    # plt.savefig("batch-score-compare.pdf")
    plt.show()
    plt.close()
    return df


def tolist(res):
    data = []
    for i in res:
        if len(i) >= 1:
            for j in i:
                data.append(j)
    return data


def mean_values(names, real_names, path=None, batch=None):
    ps = []
    for p, rn in zip(names, real_names):
        result = values(p, path, batch=batch, mu=True)
        ps.append(result)
    return tolist(ps)


def Accuracy(pred, res, sf=0.05):
    correct = 0
    for i in range(res.shape[-1]):
        # correct += ((pred[:, i].int() == res[:, i].int()).sum().item()) / res.shape[0]
        cond = torch.logical_and(
            pred[:, i].int() <= res[:, i].int() * (1 + sf),
            pred[:, i].int() >= res[:, i].int() * (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


def obtain_score(ics, scale=0.15, nsamples=10):
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(input_data, nsamples)
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    model = optimize.DeterministicModel(make, names, ics)
    with torch.no_grad():
        pred = model(data, cycles, types, control)
    return Accuracy(pred, results)


def cook_ics(pvs):
    data = []
    for i in range(5):
        data.append(torch.tensor(pvs[i]))
    data += [torch.tensor([pvs[5], pvs[6], pvs[7]])]
    data += [torch.tensor([pvs[8], pvs[9], pvs[10]])]
    return data


def calculate_loss(path=None, batch=10):
    losses = []
    for i in range(10):
        df = read_file(path, batch, i)
        losses.append(df.to_numpy().flatten())
    loss_hist = np.vstack(losses).T
    return loss_hist.mean(axis=1) / batch


def show_values(axs, vdata, orient="v", space=0.01, vself=False):
    def _single(ax):
        if orient == "v":
            if vself:
                for p, v in zip(ax.patches, vdata):
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = "{:.2f}".format(v)
                    ax.text(_x, _y, value, ha="center")
            else:
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = "{:.2f}".format(p.get_height())
                    ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            if vself:
                for p, v in zip(ax.patches, vdata):
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                    value = "{:.2f}".format(v)
                    ax.text(_x, _y, value, ha="left")
            else:
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                    value = "{:.2f}".format(p.get_width())
                    ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def display_figures(ax, show):
    i = 0
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            value = show[i]
            ax.text(p.get_x() + p.get_width() / 2, h + 10, value, ha="center")
            i = i + 1


def optim():

    df = {
        "params": [0.44, 0.49, 0.03, 0.38, 0.36],
        "names": [r"$n$", r"$\eta$", r"$s_{0}$", r"$A$", r"$p$"],
        "actual": [6.244, 294.6, 7.7, 313.6, 0.374],
    }

    sns.set(font_scale=1.5, style="white")
    p = sns.barplot(data=df, x="names", y="params")
    plt.legend(loc="upper right", frameon=False, ncol=1, prop={"size": 20})
    ax = plt.gca()
    plt.ylim(0, 0.6)
    plt.xlabel("", fontsize=23)
    plt.ylabel("Value", fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    show_values(p, df["actual"], space=0, vself=True)
    plt.tight_layout()
    plt.savefig("powerlaw-params.pdf")
    plt.show()
    plt.close()
    return df


if __name__ == "__main__":

    """
    _ = optim()
    sys.exit("stop")
    """

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/powerlaw/deterministic/"
    bspath = path1 + path2

    df_loss = read_file(bspath, "loss-batch")
    df_time = read_file(bspath, "time-batch")

    plt.plot(df_loss, lw=3)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=23)
    plt.ylabel("Loss", fontsize=23)
    # plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("powerlaw-loss-II.pdf")
    plt.show()
    plt.close()
