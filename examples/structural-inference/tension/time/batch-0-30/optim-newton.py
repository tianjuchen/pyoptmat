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
import time
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


    start_time = time.time()
    time_list = [start_time]
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.00
    nsamples = 30  # at each strain rate
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
    ics = torch.tensor([0.1 for i in range(len(names))], device=device)

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    niter = 10
    lr = 1.0e-2
    optim = torch.optim.LBFGS(model.parameters())

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
        for name, n in zip(names, model.parameters()):
            print("%s:" % (name), n.data, n.grad)

        end_time = time.time()
        use_time = end_time - start_time
        time_list.append(use_time)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

    # 8) Plot the convergence history
    np.savetxt("loss-newton.txt", loss_history)
    np.savetxt("time-newton.txt", time_list)

