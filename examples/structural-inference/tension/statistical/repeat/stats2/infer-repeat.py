#!/usr/bin/env python3

"""
    Tutorial example of training a statistical model to tension test data
    from from a known distribution.
"""

import sys
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import numpy.random as ra
import numpy as np
import xarray as xr
import torch

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import tqdm

import pyro
from pyro.infer import SVI
import pyro.optim as optim

import matplotlib.pyplot as plt
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    """
        Maker with Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


def train(ik, scale=0.15, nsamples=30):

    n_mu = []
    eta_mu = []
    s0_mu = []
    R_mu = []
    d_mu = []

    n_std = []
    eta_std = []
    s0_std = []
    R_std = []
    d_std = []

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the priors
    names = ["n", "eta", "s0", "R", "d"]
    loc_loc_priors = [
        torch.tensor(ra.random(), device=device) for i in range(len(names))
    ]
    loc_scale_priors = [torch.tensor(0.15, device=device) for i in range(len(names))]
    scale_scale_priors = [torch.tensor(0.15, device=device) for i in range(len(names))]

    eps = torch.tensor(1.0e-4, device=device)

    print("Initial parameter values:")
    print("\tloc loc\t\tloc scale\tscale scale")
    for n, llp, lsp, sp in zip(
        names, loc_loc_priors, loc_scale_priors, scale_scale_priors
    ):
        print("%s:\t%3.2f\t\t%3.2f\t\t%3.2f" % (n, llp, lsp, sp))
    print("")

    # 3) Create the actual model
    model = optimize.HierarchicalStatisticalModel(
        make, names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps
    ).to(device)

    # 4) Get the guide
    guide = model.make_guide()

    # 5) Setup the optimizer and loss
    lr = 1.0e-2
    g = 1.0
    niter = 200
    lrd = g ** (1.0 / niter)
    num_samples = 1

    optimizer = optim.Adam({"lr": lr})
    #optimizer = optim.ClippedAdam({"lr": lr, "lrd": lrd})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = SVI(model, guide, optimizer, loss=ls)

    # 6) Infer!
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data, cycles, types, control, results)
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)
    
    # 7) Print out results
    print("")
    print("Inferred distributions:")
    print("\tloc\t\tscale")
    for n in names:
        s = pyro.param(n + model.scale_suffix + model.param_suffix).data
        m = pyro.param(n + model.loc_suffix + model.param_suffix).data
        print("%s:\t%3.2f/0.50\t%3.2f/%3.2f" % (n, m, s, scale))
    print("")


    for i, n in enumerate(names):
        s = pyro.param(n + model.scale_suffix + model.param_suffix).item()
        m = pyro.param(n + model.loc_suffix + model.param_suffix).item()
        if i == 0:
            n_mu.append(m)
            n_std.append(s)
        elif i == 1:
            eta_mu.append(m)
            eta_std.append(s)
        elif i == 2:
            s0_mu.append(m)
            s0_std.append(s)
        elif i == 3:
            R_mu.append(m)
            R_std.append(s)
        elif i == 4:
            d_mu.append(m)
            d_std.append(s)

    # 8) Plot convergence
    np.savetxt("loss-history-{}-{}.txt".format(scale, ik), loss_hist)

    return n_mu, eta_mu, s0_mu, R_mu, d_mu, n_std, eta_std, s0_std, R_std, d_std

if __name__ == "__main__":

    for ik in range(10):
        n_mu, eta_mu, s0_mu, R_mu, d_mu, n_std, eta_std, s0_std, R_std, d_std = train(ik)

    data = pd.DataFrame({
        "n_mu": n_mu,
        "n_std": n_std,
        "eta_mu": eta_mu,
        "eta_std": eta_std,
        "s0_mu": s0_mu,
        "s0_std": s0_std,
        "R_mu": R_mu,
        "R_std": R_std,
        "d_mu": d_mu,
        "d_std": d_std,
    }) 
        
    data.to_csv('res_2.csv')
    
