#!/usr/bin/env python3

"""
    Helper functions for the structural material model inference with
    tension tests examples.
"""

import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
from pyoptmat import models, flowrules, hardening, optimize
from pyoptmat.temperature import ConstantParameter as CP
import matplotlib.pyplot as plt
from tqdm import tqdm

import tqdm

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
C_true = 150000.0

# Scale factor used in the model definition
sf = 0.5

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def make_model(C, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """
    
    model = models.SimpleelasticModel(CP(
        C,
        scaling=optimize.bounded_scale_function(
            (
                torch.tensor(C_true * (1 - sf), device=device),
                torch.tensor(C_true * (1 + sf), device=device),
            )
            ),
        ))

    return models.ModelIntegrator(model, use_adjoint=False, **kwargs)


def generate_input(erates, emax, ntime):
    """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
    """
    strain = torch.repeat_interleave(
        torch.linspace(0, emax, ntime, device=device)[None, :], len(erates), 0
    ).T.to(device)
    time = strain / erates

    return time, strain


def downsample(rawdata, nkeep, nrates, nsamples):
    """
    Return fewer than the whole number of samples for each strain rate
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )


if __name__ == "__main__":
    # Running this script will regenerate the data
    ntime = 200
    emax = 0.5
    erates = torch.logspace(-2, -8, 7, device=device)
    
    nrates = len(erates)
    nsamples = 50

    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    times, strains = generate_input(erates, emax, ntime)
    
    """
    temperature = torch.zeros_like(strains)
    model = make_model(torch.tensor(150000.0, device=device))
    with torch.no_grad():
        stresses = model.solve_elastic(
            times, strains, temperature
        )[:, :, 0]

    plt.plot(strains.numpy(), stresses.numpy())
    plt.show()

    sys.exit("stop")
    """
    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, nrates, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)
        full_temperatures = torch.zeros_like(full_strains)

        for i in tqdm.tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(ra.normal(0.5, scale), device=device),
            )

            with torch.no_grad():
                full_stresses[:, :, i] = model.solve_elastic(
                    times, strains, full_temperatures[:, :, i]
                )[:, :, 0]

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["tensile"] * (nsamples * len(erates)))
        controls = np.array(["strain"] * (nsamples * len(erates)))

        ds = xr.Dataset(
            {
                "time": (["ntime", "nexp"], full_times.flatten(-2, -1).cpu().numpy()),
                "strain": (
                    ["ntime", "nexp"],
                    full_strains.flatten(-2, -1).cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "nexp"],
                    full_stresses.flatten(-2, -1).cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "nexp"],
                    full_temperatures.cpu().flatten(-2, -1).numpy(),
                ),
                "cycle": (["ntime", "nexp"], full_cycles.flatten(-2, -1).cpu().numpy()),
                "type": (["nexp"], types),
                "control": (["nexp"], controls),
            },
            attrs={"scale": scale, "nrates": nrates, "nsamples": nsamples},
        )

        ds.to_netcdf("scale-%3.2f.nc" % scale)
