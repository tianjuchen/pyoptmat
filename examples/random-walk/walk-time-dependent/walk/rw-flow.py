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

import xarray as xr
import torch

from maker import make_model, load_subset_data

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
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


def smooth_time(data):
    new_data = torch.empty_like(data)
    for i in range(data.shape[-1]):
        new_data[0, :, i] = torch.linspace(0, data[0, -1, i], data.shape[1])
    new_data[1] = data[1]
    new_data[2] = data[2]
    return new_data


def comparison(ics, figure_name):

    scale = 0.00
    nsamples = 1  # 20 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "test-scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    # visualize
    names = ["n", "eta", "s0", "R", "d", "C", "g"]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
    print("")

    model = optimize.DeterministicModel(make, names, ics)

    with torch.no_grad():
        pred = model(data, cycles, types, control)
    plt.plot(
        data[2].detach().numpy(), pred.detach().numpy(), "r-", lw=3, label="initial"
    )
    plt.plot(
        data[2].detach().numpy(), results.detach().numpy(), "k--", lw=3, label="actual"
    )
    plt.xlabel("Strain (mm/mm)", fontsize=18)
    plt.ylabel("Stress (MPa)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.legend(prop={"size": 16}, frameon=False)
    # plt.rcParams.update({'font.size': 36})
    # plt.savefig("{}.pdf".format(figure_name))
    plt.show()
    plt.close()
    return names


if __name__ == "__main__":

    ics = [
        torch.tensor(0.4230, device=device),
        torch.tensor(0.3415, device=device),
        torch.tensor(0.6544, device=device),
        torch.tensor(0.5891, device=device),
        torch.tensor(0.7257, device=device),
        torch.tensor([0.5545, 0.5692, 0.2745], device=device),
        torch.tensor([0.2661, 0.7073, 0.4527], device=device),
    ]

    ics = [
        torch.tensor(0.4368, device=device),
        torch.tensor(0.4507, device=device),
        torch.tensor(0.7640, device=device),
        torch.tensor(0.5068, device=device),
        torch.tensor(0.5049, device=device),
        torch.tensor([0.5003, 0.5211, 0.4706], device=device),
        torch.tensor([0.3565, 0.7994, 0.4607], device=device),
    ]

    _ = comparison(ics, "initial")
