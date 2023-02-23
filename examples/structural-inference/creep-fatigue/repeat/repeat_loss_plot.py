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


def read_file(path, nsample, indice=0):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        batch = os.path.basename(f).split("-")[2].split("-")[0]
        index = os.path.basename(f).split("-")[3].split(".txt")[0]
        if float(batch) == nsample and float(index) == indice:
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
            pred[:, i].int() <= res[:, i].int() * (1+sf),
            pred[:, i].int() >= res[:, i].int() * (1-sf))
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
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples
    )    
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


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/repeat/"
    bspath = path1 + path2

    path_b1 = bspath + "repeat/"
    path_b2 = bspath + "repeat2/"
    path_b3 = bspath + "repeat3/"
    path_b4 = bspath + "repeat4/"
    path_b5 = bspath + "repeat5/"

    paths = [path_b1, path_b2, path_b3, path_b4, path_b5]

    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    real_names = [
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

    loss_10 = calculate_loss(path=path_b1, batch=10)
    loss_20 = calculate_loss(path=path_b2, batch=20)
    loss_30_1 = calculate_loss(path=path_b1, batch=30)
    loss_30_2 = calculate_loss(path=path_b2, batch=30)
    loss_30_3 = calculate_loss(path=path_b3, batch=30)
    loss_30 = np.vstack((loss_30_1, loss_30_2, loss_30_3)).T.mean(axis=1)
    
    plt.plot(loss_10, lw=3, label=r"$n_{sample}$=10")
    plt.plot(loss_20, lw=3, label=r"$n_{sample}$=20")
    # plt.plot(loss_30, lw=3, label=r"$n_{sample}$=30")

    ax = plt.gca()
    plt.xlabel("Step", fontsize=23)
    plt.ylabel("Loss", fontsize=23)
    plt.legend(loc="best", ncol=1, prop={"size": 23}, frameon=False)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("cyclic-loss-II.pdf")
    plt.show()
    plt.close()