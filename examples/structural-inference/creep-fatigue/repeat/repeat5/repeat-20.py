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

from maker import make_model, load_subset_data
import pandas as pd
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


def train(nsamples, niter=300, repeat=None):
    ivalues = []
    pvalues = []

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale, lr = 0.15, 1.0e-2
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    ics = [torch.tensor(ra.uniform(0, 1), device=device) for i in range(5)] + [
        torch.tensor(ra.uniform(0, 1, size=(3,)), device=device),
        torch.tensor(ra.uniform(0, 1, size=(3,)), device=device),
    ]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
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
    print("Optimized parameter accuracy (target values are all 0.5):")
    for n in names:
        print(("%s:\t" % n) + str(getattr(model, n).data))
        pvalues.append(getattr(model, n).data)

    # 8) Save the convergence history
    np.savetxt("loss-history-{}-{}.txt".format(nsamples, repeat), loss_history)

    return ivalues, pvalues


def flatten_list(values):
    data = []
    for i in values:
        if i.dim() > 0:
            for j in i:
                data.append(j.item())
        else:
            data.append(i.item())
    return data

if __name__ == "__main__":

    
    nrepeat, batch = 10, 20
    pns = []
    petas = []
    ps0s = []
    pRs = []
    pds = []
    pC1s = []
    pC2s = []
    pC3s = []
    pg1s = []
    pg2s = []
    pg3s = []


    ins = []
    ietas = []
    is0s = []
    iRs = []
    ids = []
    iC1s = []
    iC2s = []
    iC3s = []
    ig1s = []
    ig2s = []
    ig3s = []

    for i in range(nrepeat):
        ics, fcs = train(batch, repeat=i)
        
        ivalues = flatten_list(ics)
        pvalues = flatten_list(fcs)
        
        pns.append(pvalues[0])
        petas.append(pvalues[1])
        ps0s.append(pvalues[2])
        pRs.append(pvalues[3])
        pds.append(pvalues[4])
        pC1s.append(pvalues[5])
        pC2s.append(pvalues[6])
        pC3s.append(pvalues[7])
        pg1s.append(pvalues[8])
        pg2s.append(pvalues[9])
        pg3s.append(pvalues[10])

        ins.append(ivalues[0])
        ietas.append(ivalues[1])
        is0s.append(ivalues[2])
        iRs.append(ivalues[3])
        ids.append(ivalues[4])
        iC1s.append(ivalues[5])
        iC2s.append(ivalues[6])
        iC3s.append(ivalues[7])
        ig1s.append(ivalues[8])
        ig2s.append(ivalues[9])
        ig3s.append(ivalues[10])
    
    pdata = pd.DataFrame(
        {
            "n": np.array(pns),
            "eta": np.array(petas),
            "s0": np.array(ps0s),
            "R": np.array(pRs),
            "d": np.array(pds),
            "C1": np.array(pC1s),
            "C2": np.array(pC2s),
            "C3": np.array(pC3s),
            "g1": np.array(pg1s),
            "g2": np.array(pg2s),
            "g3": np.array(pg3s),
            
        }
    )
    pdata.to_csv("pvalues-{}-{}.csv".format(nrepeat, batch))

    idata = pd.DataFrame(
        {
            "n": np.array(ins),
            "eta": np.array(ietas),
            "s0": np.array(is0s),
            "R": np.array(iRs),
            "d": np.array(ids),
            "C1": np.array(iC1s),
            "C2": np.array(iC2s),
            "C3": np.array(iC3s),
            "g1": np.array(ig1s),
            "g2": np.array(ig2s),
            "g3": np.array(ig3s),
            
        }
    )
    idata.to_csv("ivalues-{}-{}.csv".format(nrepeat, batch))


