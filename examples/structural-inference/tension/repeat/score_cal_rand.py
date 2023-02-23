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
from pyoptmat import optimize, experiments, utility
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
device = torch.device("cpu")

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


def split(data, results, cycles, types, control, test_percent=0.2):
    test_inds = []

    inds = torch.arange(data.shape[-1]).cpu().numpy()
    test_inds.extend(
        ra.choice(inds, size=int(np.ceil(len(inds) * test_percent)), replace=False)
    )
    test_inds.sort()
    train_inds = sorted(list(set(list(range(len(types)))) - set(test_inds)))

    test_data = data[:, :, test_inds]
    test_results = results[:, test_inds]
    test_cycles = cycles[:, test_inds]
    test_types = types[test_inds]
    test_control = control[test_inds]

    train_data = data[:, :, train_inds]
    train_results = results[:, train_inds]
    train_cycles = cycles[:, train_inds]
    train_types = types[train_inds]
    train_control = control[train_inds]

    return (
        test_data,
        test_results,
        test_cycles,
        test_types,
        test_control,
        train_data,
        train_results,
        train_cycles,
        train_types,
        train_control,
    )


def Accuracy(pred, res, sf=0.00):
    correct = 0
    for i in range(res.shape[-1]):
        # correct += ((pred[:, i].int() == res[:, i].int()).sum().item()) / res.shape[0]
        cond = torch.logical_and(
            pred[:, i] <= res[:, i] * (1 + sf),
            pred[:, i] >= res[:, i] * (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def New_Accuracy(pred, res, sf=0.15, tiny=torch.tensor(1.0e-10)):
    pred = (pred + tiny) / (res + tiny)
    correct = 0
    for i in range(res.shape[-1]):
        cond = torch.logical_and(
            pred[:, i] <= (1 + sf),
            pred[:, i] >= (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def visualize_variance(strain, stress_true, stress_calc, nsamples, alpha=0.05):
    """
    Visualize variance for batched examples

    Args:
      strain (torch.tensor):        input strain
      stress_true (torch.tensor):   actual stress values
      stress_calc (torch.tensor):   simulated stress values

    Keyword Args:
      alpha (float): alpha value for shading
    """

    ax = plt.gca()
    plt.plot(strain.numpy(), stress_true.numpy(), "-", lw=3)

    plt.plot(strain.numpy(), stress_calc.numpy(), "k-", lw=5)

    plt.xlabel("Strain (mm/mm)", fontsize=18)
    plt.ylabel("Stress (MPa)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    plt.savefig("batch-stress-strain-{}.pdf".format(nsamples))
    plt.show()
    plt.close()
    return stress_true


if __name__ == "__main__":

    scale = 0.00
    nsamples = 50  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    true_data, true_results, true_cycles, true_types, true_control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    indices = torch.randint(1, 50, (30,))
    # indices = torch.randperm(30)
    samples = [10, 20, 30]

    for nsam in samples:
        # 1) Load the data for the variance of interest,
        #    cut down to some number of samples, and flatten
        scale = 0.15
        nsamples = 50  # at each strain rate
        input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
        data, results, cycles, types, control = downsample(
            experiments.load_results(input_data, device=device),
            nsamples,
            input_data.nrates,
            input_data.nsamples,
        )

        # 2) Setup names for each parameter and the initial conditions
        names = ["n", "eta", "s0", "R", "d"]
        if nsam == 10:
            ics = torch.tensor(
                [0.49848774, 0.52056504, 0.47528703, 0.51003132, 0.45022538],
                device=device,
            )
        elif nsam == 20:
            ics = torch.tensor(
                [0.51255359, 0.50202772, 0.50684044, 0.51145004, 0.51277785],
                device=device,
            )
        elif nsam == 30:
            ics = torch.tensor(
                [0.48810693, 0.51449917, 0.51388876, 0.52200991, 0.51014777],
                device=device,
            )
        print("Initial parameter values:")
        for n, ic in zip(names, ics):
            print("%s:\t%3.2f" % (n, ic))
        print("")

        # 3) Create the actual model
        model = optimize.DeterministicModel(make, names, ics)

        with torch.no_grad():
            pred = model(data, cycles, types, control)

        # _ = visualize_variance(data[2], results, pred, nsamples)
        # sys.exit("stop")
        # score = Accuracy(pred.index_select(1, indices), results.index_select(1, indices))
        score = New_Accuracy(pred.index_select(1, indices), results.index_select(1, indices))
        print("Batch of %s  with score is %3.2f" % (nsam, score))
