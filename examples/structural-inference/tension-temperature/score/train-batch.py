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


def Accuracy(pred, res):
    correct = 0
    for i in range(res.shape[-1]):
        correct += (
            (pred[:, i].int() == res[:, i].int())
            .sum()
            .item()
        ) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 30  # at each strain rate
    percent = 0.2
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    (
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
    ) = split(data, results, cycles, types, control, test_percent=percent)

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d"]
    ics = torch.tensor([ra.uniform(0, 1) for i in range(len(names))], device=device)

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    niter, lr = 200, 0.01
    optim = torch.optim.Adam(model.parameters())

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(train_data, train_cycles, train_types, train_control)
        lossv = loss(
            pred,
            train_results,
        )
        lossv.backward()
        return lossv

    train_score = []
    test_score = []
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    train_history = []
    test_history = []
    for i in t:
        pred = model(train_data, train_cycles, train_types, train_control)
        closs = optim.step(closure)
        total_train_accuracy = Accuracy(pred, train_results)
        train_score.append(total_train_accuracy)
        train_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %4.3e" % train_history[-1])
        np.savetxt("train-history-{}.txt".format(percent), train_history)
        
        with torch.no_grad():
            res_test = model(test_data, test_cycles, test_types, test_control)
            test_history.append(loss(res_test, test_results).detach().cpu().numpy())
            np.savetxt("test-history-{}.txt".format(percent), test_history)
            total_valid_accuracy = Accuracy(res_test, test_results)

        test_score.append(total_valid_accuracy)
        print(f"Training acc for epoch {i}: {total_train_accuracy}")
        print(f"Validation acc for epoch {i}: {total_valid_accuracy}")

        for name, n in zip(names, model.parameters()):
            print("%s:" % (name), n.data, n.grad)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

    print(train_score)
    print(test_score)

    np.savetxt("train-score-{}.txt".format(percent), train_score)
    np.savetxt("test-score-{}.txt".format(percent), test_score)
