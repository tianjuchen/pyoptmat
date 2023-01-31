#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path
import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_subset_data

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
device = torch.device(dev)


def sin_wave(data, amp, N_min=5, N_max=8):

    new_data = torch.empty_like(data)

    for i in range(data.shape[-1]):
        new_data[0, :, i] = torch.linspace(0, data[0, -1, i], data.shape[1])
        N = ra.randint(N_min, N_max)
        period = data[0, -1, i] / N
        new_data[2, :, i] = amp * torch.sin(2 * np.pi / period * new_data[0, :, i])

    new_data[1] = data[1]

    return new_data


def sin_strain(x, amp, N_min=5, N_max=8):
    N = ra.randint(N_min, N_max)
    period = x[-1] / N
    strain = amp * np.sin(2 * np.pi / period * x)
    return strain




if __name__ == "__main__":

    # Running this script will regenerate the data

    # Maximum strain in the cycle
    max_strains = np.logspace(np.log10(0.002), np.log10(0.02), 5)

    # Tension hold in the cycle
    tension_holds = np.array([1e-6, 60.0, 5 * 60.0, 60 * 60.0])

    # The data will sample all combinations of max_strains and
    # tension_holds, with the following fixed parameters
    R = -1.0  # Fully reversed loading
    strain_rate = 1.0e-3  # Fixed strain rate
    compression_hold = 1.0e-6  # No compression hold
    temperature = 500.0  # Problem will be temperature independent

    # Discretization
    N = 5  # Number of cycles
    nsamples = 20  # Number of repeats of each test
    nload = 20  # Time steps during the load period of each test
    nhold = 20  # Time steps during the hold period of each test

    # Scale values to generate
    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    # Generate the input data for a *single* run through each test
    ntime = N * 2 * (2 * nload + nhold) + 1
    ntests = max_strains.shape[0] * tension_holds.shape[0]
    times = torch.zeros(ntime, ntests, device=device)
    strains = torch.zeros(ntime, ntests, device=device)
    cycles = torch.zeros(ntime, ntests, dtype=int, device=device)

    strain_range = []
    hold_times = []
    i = 0
    for max_e in max_strains:
        for t_hold in tension_holds:
            timesi, strainsi, cyclesi = experiments.sample_cycle_normalized_times(
                {
                    "max_strain": max_e / 2.0,
                    "R": R,
                    "strain_rate": strain_rate,
                    "tension_hold": t_hold,
                    "compression_hold": compression_hold,
                },
                N,
                nload,
                nhold,
            )
            times[:, i] = torch.tensor(timesi, device=device)
            wave_strain = sin_strain(timesi, max_e)
            strains[:, i] = torch.tensor(wave_strain, device=device)
            cycles[:, i] = torch.tensor(cyclesi, device=device)
            i += 1
            strain_range.append(max_e)
            hold_times.append(t_hold)

    temperatures = torch.ones_like(times) * temperature

    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, ntests, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)
        full_temperatures = torch.zeros_like(full_strains)
        full_cycles = torch.zeros_like(full_times, dtype=int)
        full_ranges = torch.empty(full_times.shape[1:])
        full_rates = torch.empty_like(full_ranges)
        full_holds = torch.empty_like(full_ranges)

        for i in tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains
            full_temperatures[:, :, i] = temperatures
            full_cycles[:, :, i] = cycles
            full_ranges[:, i] = torch.tensor(strain_range, device=device)
            full_rates[:, i] = strain_rate
            full_holds[:, i] = torch.tensor(hold_times, device=device)

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(0.5, device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale, size=(3,)), device=device),
                torch.tensor(ra.normal(0.5, scale, size=(3,)), device=device),
                device=device,
            )

            with torch.no_grad():
                full_stresses[:, :, i] = model.solve_strain(
                    times, strains, temperatures
                )[:, :, 0]

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["direct_data"] * (nsamples * ntests)).reshape(
            ntests, nsamples
        )
        controls = np.array(["strain"] * (nsamples * ntests)).reshape(ntests, nsamples)

        # This example uses a nonstandard format that you can read with `experiments.load_results`
        # but makes it easer to downsample the data.
        # Use the load_subset_data function above to read in the data
        ds = xr.Dataset(
            {
                "time": (["ntime", "ntests", "nsamples"], full_times.cpu().numpy()),
                "strain": (
                    ["ntime", "ntests", "nsamples"],
                    full_strains.cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "ntests", "nsamples"],
                    full_stresses.cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "ntests", "nsamples"],
                    full_temperatures.cpu().numpy(),
                ),
                "cycle": (["ntime", "ntests", "nsamples"], full_cycles.cpu().numpy()),
                "type": (["ntests", "nsamples"], types),
                "control": (["ntests", "nsamples"], controls),
                "strain_ranges": (["ntests", "nsamples"], full_ranges.numpy()),
                "strain_rates": (["ntests", "nsamples"], full_rates.numpy()),
                "hold_times": (["ntests", "nsamples"], full_holds.numpy()),
            },
            attrs={"scale": scale},
        )

        ds.to_netcdf("sin-scale-%3.2f.nc" % scale)
