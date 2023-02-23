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

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import tqdm

import matplotlib.pyplot as plt

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
def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


def fn_scaling(x_input, maxv, minv, tiny=torch.tensor(1.0e-10)):
    return (x_input - minv) / (maxv - minv + tiny)


# def fn_scaling(x_input, maxv, minv, tiny=torch.tensor(1.0e-10)):
    # return torch.mean((x_input - minv) / (maxv - minv + tiny), 1)


if __name__ == "__main__":

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 5  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    print(data.shape)
    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d"]
    ics = torch.tensor([0.46, 0.44, 0.31, 0.44, 0.49], device=device)
    # ics = torch.tensor([0.46, 0.47, 0.44, 0.46, 0.47], device=device)

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)
    with torch.no_grad():
        pred_results = model(data, cycles, types, control)

    scaleres, factor, savefigure, display = False, 1.0, False, True

    if scaleres:
        maxv = torch.amax(results, 0)
        minv = torch.amin(results, 0)
        results = fn_scaling(results, maxv, minv)
        pred_results = fn_scaling(pred_results, maxv, minv)


    plt.plot(data[2].numpy(), torch.mean(pred_results, 1).numpy(), '-', lw=3)
    plt.plot(data[2].numpy(), torch.mean(results, 1).numpy(), 'k--', lw=3)
    plt.show()
    plt.close()
    sys.exit("stop")

    ax_limit = torch.max(results.max(), pred_results.max()).detach().numpy()
    fig, ax = plt.subplots()
    xs = np.linspace(
        0,
        factor * ax_limit,
        100,
    )
    ax.plot(
        results.detach().numpy(),
        pred_results.detach().numpy(),
        "o",
        # color="lightskyblue",
        markersize=5,
        markevery=1,
        alpha=0.75,
    )
    ax.plot(xs, xs, "k--", lw=4)
    """
    ax.fill_between(xs, xs*(1-0.05),
        xs*(1+0.05), alpha=0.5, color="lightskyblue")
    """
    plt.xlim([0, factor * ax_limit])
    plt.ylim([0, factor * ax_limit])
    # plt.legend()
    plt.xlabel("Actual", fontsize=18)
    plt.ylabel("Prediction", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    if savefigure:
        plt.savefig("Comparison-{}-{}.pdf".format(nsamples, scale))
    if display:
        plt.show()
    plt.close()
