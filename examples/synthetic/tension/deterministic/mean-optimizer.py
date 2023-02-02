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


def mean_opt(icss, optimizers, batch, savefigure=False, display=False):

    for i, (ics, optimizer, nsamples) in enumerate(zip(icss, optimizers, batch)):
        scale = 0.15
        exp_times, exp_strains, exp_temperatures, exp_results = load_data(
            scale, nsamples, device=device
        )
        # define the strain rate index
        inds = torch.arange(7)
        for ind in inds:
            times = extract(exp_times, ind)
            strains = extract(exp_strains, ind)
            temperatures = extract(exp_temperatures, ind)
            results = extract(exp_results, ind)

            names = ["n", "eta", "s0", "R", "d"]
            print("Initial parameter values:")
            for n, ic in zip(names, ics):
                print("%s:\t%3.2f" % (n, ic))
            print("")

            model = optimize.DeterministicModel(make, names, ics)
            with torch.no_grad():
                pred_results = model(times, strains, temperatures)

            if i != 3:
                plt.plot(
                    strains[:, 0].numpy(),
                    torch.mean(results, 1).numpy(),
                    "-",
                    color="#ff7f0e",
                    lw=3,
                    alpha=0.65,
                )
                if optimizer == "adam":
                    plt.plot(
                        strains[:, 0].numpy(),
                        torch.mean(pred_results, 1).numpy(),
                        "-",
                        color="#1f77b4",
                        lw=3,
                        alpha=0.75,
                    )
                else:
                    plt.plot(
                        strains[:, 0].numpy(),
                        torch.mean(pred_results, 1).numpy(),
                        "-",
                        color="#2ca02c",
                        lw=3,
                        alpha=0.75,
                    )

            else:
                plt.plot(
                    strains[:, 0].numpy(),
                    torch.mean(results, 1).numpy(),
                    "-",
                    color="#ff7f0e",
                    lw=3,
                    alpha=0.65,
                )
                if optimizer == "adam":
                    plt.plot(
                        strains[:, 0].numpy(),
                        torch.mean(pred_results, 1).numpy(),
                        "-",
                        color="#1f77b4",
                        lw=3,
                        alpha=0.75,
                    )
                else:
                    plt.plot(
                        strains[:, 0].numpy(),
                        torch.mean(pred_results, 1).numpy(),
                        "-",
                        color="#2ca02c",
                        lw=3,
                        alpha=0.75,
                    )
    ax = plt.gca()
    handles = [
        mpatches.Patch(
            facecolor="#1f77b4",
            label="Adam",
        ),
        mpatches.Patch(
            facecolor="#2ca02c",
            label="Newton",
        ),
        mpatches.Patch(
            facecolor="#ff7f0e",
            label="Observation",
        ),
    ]
    plt.legend(handles=handles, prop={"size": 18}, frameon=False)
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
        plt.savefig("opt.pdf")
    if display:
        plt.show()
    plt.close()
    return inds


if __name__ == "__main__":

    # adam
    ics1 = torch.tensor([0.46, 0.44, 0.31, 0.44, 0.49], device=device)  # from batch = 5
    ics2 = torch.tensor(
        [0.46, 0.47, 0.44, 0.46, 0.47], device=device
    )  # from batch = 30

    # newton
    ics3 = torch.tensor([0.33, 0.49, 0.60, 0.44, 0.40], device=device)  # from batch = 5
    ics4 = torch.tensor(
        [0.46, 0.46, 0.53, 0.44, 0.45], device=device
    )  # from batch = 30

    icss = [ics1, ics2, ics3, ics4]
    batch = np.array([5, 30, 5, 30])
    optimizers = ["adam", "newton", "adam", "newton"]

    _ = mean_opt(icss, optimizers, batch, savefigure=True, display=True)
