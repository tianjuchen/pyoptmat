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
    return make_model(C, device=device, **kwargs).to(
        device
    )

def normalize(data, m):
    return data / m

if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.00
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    print(results)
    """
    amount = 1
    data = data[:, :, :amount]
    results = results[:, :amount]
    cycles = cycles[:, :amount]
    """
    pmax = results.mean(dim=0)
    
    # 2) Setup names for each parameter and the initial conditions
    names = ["C"]
    ics = torch.tensor([ra.uniform(0, 1) for i in range(len(names))], device=device)
    
    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicHyperElasticModel(make, names, ics)

    # 4) Setup the optimizer
    niter = 100
    lr = 1.0e-2
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        with torch.autocast("cpu"):
            pred = model(data)
            lossv = loss(normalize(pred, pmax), normalize(results, pmax))
        lossv.backward()
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(0.01, 0.99)        
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
        for p in model.parameters():
            print(p.data, p.grad)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

    # 8) Save the convergence history
    np.savetxt("loss-history-{}.txt".format(scale), loss_history)
