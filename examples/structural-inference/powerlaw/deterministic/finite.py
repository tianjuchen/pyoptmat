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
from scipy import optimize as opt
from maker import make_model, downsample, power_model

from pyoptmat import optimize, experiments
from tqdm import tqdm
import time
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

res_values = []

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, A, p, scale_fn, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return power_model(
        torch.tensor(0.5),
        n,
        eta,
        s0,
        A,
        p,
        scale_functions=scale_fn,
        device=device,
        **kwargs
    ).to(device)



def residual(yobs, ytest):
    likelihood = 0.0
    for i in range(yobs.shape[-1]):
        likelihood += np.sum((yobs.numpy()[:, i] - ytest.numpy()[:, i])**2)
    return likelihood
    

def train(p, nsamples=1, scale=0.00):
    pvalues = []
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "A", "p"]
    ics = [torch.tensor(i, device=device)  for i in p]
    
    actual_maker = lambda *x, **kwargs: make(
        *x, None, use_adjoint=True, miter=10, **kwargs
    )
    
    # 3) Create the actual model
    model = optimize.DeterministicModel(actual_maker, names, ics)
    with torch.no_grad():
        pred = model(data, cycles, types, control)
    
    res = residual(pred, results)
    res_values.append(res)
    print("current residual is %3.2f:" %(res))

    return res


def param_range(prange):
    var_ranges = []
    for i in range(len(prange)):
        var_ranges.append(prange[i])
    return var_ranges

if __name__ == "__main__":

    start = time.time()
    initial_range = np.array(
        [
            [6.0, 8.0],  # n
            [100, 400.0],  # eta
            [50.0, 100.0],  # s0
            [10.0, 300.0],  # A
            [0.1, 0.8],  # p
        ]
    )

    var_range = param_range(initial_range)
    
    p0 = [ra.uniform(*i) for i in var_range]
    
    res = opt.minimize(
        train,
        p0,
        bounds=var_range,
        # method="L-BFGS-B",
        # parallel={"max_workers": nodenum},
    )
    print(res.success)
    print(res.x)

    end = time.time()
    print("required time is %.2f" % (end - start))


    # 8) Save the convergence history
    np.savetxt("residual-history.txt", res_values)
