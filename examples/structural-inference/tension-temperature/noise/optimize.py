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

from maker import make_model, downsample, grid_model

from pyoptmat import optimize, experiments, scaling
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

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0


def pgrid(p, T, Q=-100.0):
    return p * torch.exp(-Q/T)

Tcontrol = torch.tensor([25.0, 200.0, 300.0, 400.0, 500.0, 600.0]) + 273.15
trues = [E_true, n_true, eta_true, s0_true, R_true, d_true]
scale_fn = [
    scaling.BoundedScalingFunction(pgrid(p, Tcontrol) * 0.5, pgrid(p, Tcontrol) * 1.5)
    for p in trues
]    

if __name__ == "__main__":

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.00
    nsamples = 1  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    print("temperature levels are:", data[1, 0, :])
    # 2) Setup names for each parameter and the initial conditions
    names = ["E", "n", "eta", "s0", "R", "d"]

    ics = [torch.tensor(ra.uniform(0.01, 0.99, size=(len(Tcontrol), ))) for i in range(len(names))]
    
    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:" % (n),  ic)
    print("")


    actual_maker = lambda *x, **kwargs: grid_model(
        *x, scale_functions=scale_fn,
        Tcontrol=Tcontrol,
        use_adjoint=True, miter=10, **kwargs
    )

    # 3) Create the actual model
    model = optimize.DeterministicModel(actual_maker, names, ics)

    # 4) Setup the optimizer
    niter = 10
    optim = torch.optim.LBFGS(model.parameters())

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
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
        for param in model.parameters():
            print(param.data, param.grad)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

    # 8) Plot the convergence history
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
