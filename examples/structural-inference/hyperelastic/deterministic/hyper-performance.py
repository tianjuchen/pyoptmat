#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import glob
import pandas as pd
import os.path
import numpy.random as ra
import numpy as np
import xarray as xr
import torch
from maker import make_model, downsample
from pyoptmat import optimize, experiments
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Don't care if integration fails
import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)
# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(C, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(C, device=device, **kwargs).to(device)


def normalize(data, m):
    return data / m


def read_file(path, file_name):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        fn = os.path.basename(f).split(".txt")[0]
        if fn == file_name:
            df = pd.read_csv(f)
            return df


def ploss(path, fn):
    df = read_file(path, fn)
    print(np.arange(len(df)))
    plt.plot(np.arange(len(df)), df, lw=4)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=23)
    plt.ylabel("Loss", fontsize=23)
    # plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
    plt.locator_params(axis="both", nbins=4)
    plt.tick_params(axis="both", which="major", labelsize=23)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("hyperelastic-loss.pdf")
    plt.show()
    plt.close()
    return df


def visualize(names, ics, scale=0.00, nsamples=10, amount=0):
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    # 2) Setup names for each parameter and the initial conditions
    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")
    # 3) Create the actual model
    model = optimize.DeterministicHyperElasticModel(make, names, ics)

    with torch.no_grad():
        pred = model(data)

    plt.plot(
        data[2, :, :amount].numpy(), results[:, :amount].numpy() / 1000.0, lw=4, label="Actual"
    )
    plt.plot(
        data[2, :, :amount].numpy(),
        pred[:, :amount].numpy() / 1000.0,
        "k--",
        lw=4,
        label="Prediction",
    )
    ax = plt.gca()
    plt.xlabel("Strain", fontsize=23)
    plt.ylabel("Stress (MPa)", fontsize=23)
    # plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.locator_params(axis="y", nbins=3)
    plt.locator_params(axis="x", nbins=3)
    plt.tick_params(axis="both", which="major", labelsize=23)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("hyper-perform-{}.pdf".format(scale))
    plt.show()
    plt.close()
    return names


if __name__ == "__main__":
    """
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/structural-inference/"
    path2 = "hyperelastic/deterministic/"
    path = path1 + path2
    fn = "loss-history-0.0"
    _ = ploss(path, fn)

    sys.exit("stop")
    """
    names = ["C"]
    ics = torch.tensor(
        [ra.uniform(0.43, 0.43) for i in range(len(names))], device=device
    )

    _ = visualize(names, ics, scale=0.15, nsamples=10, amount=70)
