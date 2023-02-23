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
from maker import make_model, downsample

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
timer = []
start = time.time()
# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


def residual(yobs, ytest):
    likelihood = 0.0
    for i in range(yobs.shape[-1]):
        likelihood += np.sum((yobs.numpy()[:, i] - ytest.numpy()[:, i])**2)
    return likelihood
    

def train(p, nsamples=30, scale=0.00):
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
    names = ["n", "eta", "s0", "R", "d"]
    ics = torch.tensor([i for i in p], device=device)
    """
    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:\t%3.2f" % (n, ic))
    print("")
    """
    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)
    with torch.no_grad():
        pred = model(data, cycles, types, control)
    
    res = residual(pred, results)
    res_values.append(res)
    print("current residual is %3.2f:" %(res))

    current_time = time.time() - start
    timer.append(current_time)

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
            [0.0, 1.0],  # n
            [0.0, 1.0],  # eta
            [0.0, 1.0],  # s0
            [0.0, 1.0],  # R
            [0.0, 1.0],  # d
        ]
    )

    var_range = param_range(initial_range)
    
    #p0 = [ra.uniform(*i) for i in var_range]
    p0 = [0.1 for i in var_range]

    res = opt.minimize(
        train,
        p0,
        bounds=var_range,
        jac = "2-point",
        method="BFGS",
        # parallel={"max_workers": nodenum},
    )
    print(res.success)
    print(res.x)

    end = time.time()
    print("required time is %.2f" % (end - start))


    # 8) Save the convergence history
    np.savetxt("bfgs-2point-history.txt", res_values)
    np.savetxt("bfgs-2point-timer.txt", timer)
