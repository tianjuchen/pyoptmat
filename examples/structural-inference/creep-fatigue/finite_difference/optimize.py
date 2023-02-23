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

from maker import make_model, load_subset_data, make_model_1, make_model_2

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
device = torch.device(dev)

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model_1(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


def New_Accuracy(pred, res, sf=0.01, tiny=torch.tensor(1.0e-10)):
    pred = (pred + tiny) / (res + tiny)
    correct = 0
    for i in range(res.shape[-1]):
        cond = torch.logical_and(
            pred[:, i] <= (1 + sf),
            pred[:, i] >= (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def obtain_score(ics, scale=0.00, nsamples=20, full=False):
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples
    )    
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    model = optimize.DeterministicModel(make, names, ics)
    with torch.no_grad():
        pred = model(data, cycles, types, control)
    if full:
        return New_Accuracy(pred, results), pred, results, data
    else:
        return New_Accuracy(pred, results)


if __name__ == "__main__":


    ics = [
        torch.tensor(0.4962),
        torch.tensor(0.4984),
        torch.tensor(0.5166),
        torch.tensor(0.4795),
        torch.tensor(0.5317),
        torch.tensor([0.9702]),
        torch.tensor([0.3387]),
    ]

    score = obtain_score(ics)
    
    print("Accuracy = {}".format(score))