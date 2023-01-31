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

from maker import make_model, downsample, params_true

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
def make(g0, A, C, R, d, **kwargs):
    """
        Maker with the Young's modulus fixed
    """
    return make_model(g0, A, C, R, d, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.0
    nsamples = 1  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    names = ["g0", "A", "C", "R", "d"]
    
    scale_functions = [
            scaling.SimpleScalingFunction(torch.tensor(1.0, device = device)) for
            i in range(len(names))
    ]

    #params = torch.tensor([0.83, -4.20, -6.01, 200.41, 5.04], device=device)
    params = torch.tensor([0.60, -3.4, -5.9, 200.0, 5.0], device=device)
    model = make(*params, scale_functions=scale_functions)
    with torch.no_grad():
        pred = model.solve_strain(
                    data[0], data[2], data[1]
                )[:, :, 0]
    
    seperate = False
    
    if seperate:
        for i in range(results.shape[-1]):
            plt.plot(data[2, :, i].numpy(), pred[:, i].numpy(), label="prediction")
            plt.plot(data[2, :, i].numpy(), results[:, i].numpy(), "k--", label="experiment")
            plt.legend()
            #plt.savefig("b-flow-stress-{}.pdf".format(i))
            plt.show()
            plt.close()
    else:
        plt.plot(data[2].numpy(), pred.numpy(), label="prediction")
        #plt.plot(data[2].numpy(), results.numpy(), "k--", label="experiment")
        plt.legend()
        #plt.savefig("b-flow-stress.pdf")
        plt.show()
        plt.close()
