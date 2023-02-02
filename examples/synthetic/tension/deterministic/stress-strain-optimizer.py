#!/usr/bin/env python3

import sys

sys.path.append("../../../..")
sys.path.append("..")

import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# Run on CPU (home machine GPU is eh)
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, use_adjoint=True, device=device, **kwargs
    ).to(device)


def fn_scaling(x_input, maxv, minv, tiny=torch.tensor(1.0e-10)):
    return (x_input - minv) / (maxv - minv + tiny)


def extract(data, ind, gp=7):
    new_data = torch.empty((int(data.shape[0]), int(data.shape[-1] / gp)))
    iq = 0
    for i in range(data.shape[-1]):
        if i % gp == ind:
            new_data[:, iq] = data[:, i]
            iq += 1
    return new_data


if __name__ == "__main__":

    # true
    nsamples, scale = 1, 0.00
    ref_times, ref_strains, ref_temperatures, ref_results = load_data(
        scale, nsamples, device=device
    )
    # test
    nsamples, scale = 1, 0.15
    exp_times, exp_strains, exp_temperatures, exp_results = load_data(
        scale, nsamples, device=device
    )
    # observe
    nsamples, scale = 30, 0.15
    obs_times, obs_strains, obs_temperatures, obs_results = load_data(
        scale, nsamples, device=device
    )
    # define the strain rate index
    inds = torch.arange(7)

    for ind in inds:
        plt.plot(
            extract(obs_strains, ind).numpy(),
            extract(obs_results, ind).numpy(),
            "--",
            color="lightskyblue",
            lw=3,
            alpha=0.25,
        )

    for ind in inds:
        times = extract(exp_times, ind)
        strains = extract(exp_strains, ind)
        temperatures = extract(exp_temperatures, ind)
        results = extract(exp_results, ind)

        names = ["n", "eta", "s0", "R", "d"]

        # adam
        # ics = torch.tensor([0.46, 0.44, 0.31, 0.44, 0.49], device=device) # from batch = 5
        ics = torch.tensor([0.46, 0.47, 0.44, 0.46, 0.47], device=device) # from batch = 30

        # newton
        # ics = torch.tensor(
            # [0.33, 0.49, 0.60, 0.44, 0.40], device=device
        # )  # from batch = 5
        # ics = torch.tensor([0.46, 0.46, 0.53, 0.44, 0.45], device=device) # from batch = 30

        print("Initial parameter values:")
        for n, ic in zip(names, ics):
            print("%s:\t%3.2f" % (n, ic))
        print("")

        model = optimize.DeterministicModel(make, names, ics)
        with torch.no_grad():
            pred_results = model(times, strains, temperatures)

        plt.plot(strains.numpy(), pred_results.numpy(), "g-", lw=3, alpha=0.75)

    for ind in inds:
        plt.plot(
            extract(ref_strains, ind).numpy(),
            extract(ref_results, ind).numpy(),
            "-",
            color="tomato",
            lw=3,
            alpha=0.75,
        )

    savefigure, display = True, True
    ax = plt.gca()
    handles = [
        mpatches.Patch(
            facecolor="g",
            label="Prediction",
        ),
        mpatches.Patch(
            facecolor="lightskyblue",
            label="Observation",
        ),
        mpatches.Patch(
            facecolor="tomato",
            label="True",
        ),
    ]
    plt.legend(handles=handles, prop={"size": 16}, frameon=False)
    plt.xlabel("Strain (mm/mm)", fontsize=18)
    plt.ylabel("Stress (MPa)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    if savefigure:
        plt.savefig("adam-{}-{}.pdf".format(nsamples, scale))
    if display:
        plt.show()
    plt.close()
    sys.exit("stop")
