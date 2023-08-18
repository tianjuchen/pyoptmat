#!/usr/bin/env python3

import sys
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import xarray as xr
import torch

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on this on the cpu
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    """
        Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # Number of vectorized time steps
    time_chunk_size = 40

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # Write out the strain rate
    rates = data[2, -1, :] / data[0, -1, :]
    urates = rates.unique()
    
    for ur in urates:
        conds = rates == ur
        curr_data = data[:, :, conds]
        curr_results = results[:, conds]
        curr_cycles = cycles[:, conds]
        curr_types = types[conds]
        curr_control = control[conds]
        
        names = ["n", "eta", "s0", "R", "d"]
        sampler = optimize.StatisticalModel(
            lambda *args, **kwargs: make(*args, block_size = time_chunk_size, **kwargs),
            names,
            [1.0, 1.0, 0.0, 0.0, 0.36],
            [6.3435, 0.1549, 0.1998, 0.2480, 0.1584],
            torch.tensor(1.0e-4),
        )

        plt.figure()
        plt.plot(curr_data[2, :, :nsamples].cpu(), curr_results[:, :nsamples].cpu(), "k--")

        nsamples = 100
        alpha = 0.05 / 2


        stress_results = torch.zeros(nsamples, data.shape[1])

        for i in trange(nsamples):
            stress_results[i, :] = sampler(curr_data, curr_cycles, curr_types, curr_control)[:, 0]

        mean_result = torch.mean(stress_results, 0)
        sresults, _ = torch.sort(stress_results, 0)
        min_result = sresults[int(alpha * nsamples), :]
        max_result = sresults[int((1 - alpha) * nsamples), :]

        (l,) = plt.plot(curr_data[2, :, 0], mean_result, lw=4, color="k")
        p = plt.fill_between(curr_data[2, :, 0], min_result, max_result, alpha=0.5, color="k")

        plt.legend(
            [
                Line2D([0], [0], color="k", ls="--"),
                Line2D([0], [0], color="k", lw=4),
                Patch(facecolor="k", edgecolor=None, alpha=0.5),
            ],
            ["Synthetic data", "Model average", "Model 95% prediction interval"],
            loc="best",
        )

        plt.xlabel("Strain (mm/mm)", fontsize=27)
        plt.ylabel("Stress (MPa)", fontsize=27)
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=27)
        plt.locator_params(axis='both', nbins=3)
        plt.tick_params(axis="both", which="major", labelsize=27)
        ax = plt.gca()
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("tension-uq-%3.2f-%.2e.pdf" % (scale, ur))
        #plt.show()
        plt.close()

