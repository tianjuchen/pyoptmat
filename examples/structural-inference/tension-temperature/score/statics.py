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
import pandas as pd

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

def train(nsamples, niter=300):
    ivalues = []
    pvalues = []
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale, lr = 0.15, 1.0e-2
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d"]
    ics = torch.tensor(
        [ra.uniform(0, 1) for i in range(len(names))], device=device
    )

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
        ivalues.append(ic)
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

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

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))
        pvalues.append(getattr(model, n).data)
    
    return ivalues, pvalues


if __name__ == "__main__":

    nsamples = 5
    ni = []
    etai = []
    s0i = []
    Ri = []
    di = []

    ns = []
    etas = []
    s0s = []
    Rs = []
    ds = []
    for i in range(100):
        ivalues, pvalues = train(nsamples)
        ns.append(pvalues[0])
        etas.append(pvalues[1])
        s0s.append(pvalues[2])
        Rs.append(pvalues[3])
        ds.append(pvalues[4])
        
        ni.append(ivalues[0])
        etai.append(ivalues[1])
        s0i.append(ivalues[2])
        Ri.append(ivalues[3])
        di.append(ivalues[4])
        
    fdata = pd.DataFrame({
        "n": np.array(ns),
        "eta": np.array(etas),
        "s0": np.array(s0s),
        "R": np.array(Rs),
        "d": np.array(ds),
    }) 
    fdata.to_csv('fvalues.csv')
    
    idata = pd.DataFrame({
        "n": np.array(ni),
        "eta": np.array(etai),
        "s0": np.array(s0i),
        "R": np.array(Ri),
        "d": np.array(di),
    }) 
    idata.to_csv('ivalues.csv')
