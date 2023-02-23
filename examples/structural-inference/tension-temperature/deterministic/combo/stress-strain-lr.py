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


if __name__ == "__main__":

    lrs = [1.0e-1, 1.0e-2, 1.0e-3]
    scales = [0.00]

    ns = np.array([0.50, 0.46, 0.81])
    etas = np.array([0.50, 0.51, 0.18])
    s0s = np.array([0.50, 0.56, 0.42])
    Rs = np.array([0.50, 0.50, 0.58])
    ds = np.array([0.50, 0.49, 0.47])

    for scale in scales:
        for n, eta, s0, R, d, lr in zip(ns, etas, s0s, Rs, ds, lrs):
            # 1) Load the data for the variance of interest,
            #    cut down to some number of samples, and flatten
            nsamples = 1  # at each strain rate
            input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
            data, results, cycles, types, control = downsample(
                experiments.load_results(input_data, device=device),
                nsamples,
                input_data.nrates,
                input_data.nsamples,
            )
            # 2) Setup names for each parameter and the initial conditions
            names = ["n", "eta", "s0", "R", "d"]
            ics = torch.tensor([n, eta, s0, R, d], device=device)
            print("Initial parameter values:")
            for n, ic in zip(names, ics):
                print("%s:\t%3.2f" % (n, ic))
            print("")

            model = optimize.DeterministicModel(make, names, ics)
            pred = model(data, cycles, types, control)

            for i in range(results.shape[-1]):
                if i == results.shape[-1] - 1:
                    plt.plot(
                        data[2, :, i].detach().numpy(),
                        pred[:, i].detach().numpy(),
                        lw=3,
                        label="Prediction",
                    )
                    plt.plot(
                        data[2, :, i].detach().numpy(),
                        results[:, i].detach().numpy(),
                        "k--",
                        lw=3,
                        label="Actual",
                    )
                else:
                    plt.plot(data[2].detach().numpy(), pred.detach().numpy(), lw=3)
                    plt.plot(
                        data[2].detach().numpy(), results.detach().numpy(), "k--", lw=3
                    )

            plt.xlabel("Strain (mm/mm)", fontsize=16)
            plt.ylabel("Stress (MPa)", fontsize=16)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            # plt.title("Scattering Predictive of Stress-strain", fontsize=16)
            plt.tight_layout()
            plt.grid(False)
            plt.legend(prop={"size": 18}, frameon=False, ncol=1, loc="best")
            # plt.savefig("flow-curves-{}.pdf".format(lr))
            plt.show()
            plt.close()
