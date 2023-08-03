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
from matplotlib import RcParams

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


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)

if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 50  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    names = ["n", "eta", "s0", "R", "d"]
    sampler = optimize.StatisticalModel(
        make,
        names,
        [0.5057, 0.5044, 0.5054, 0.5161, 0.5297],
        [0.073, 0.073, 0.076, 0.126, 0.136],
        torch.tensor(1.0e-4),
    )

    plt.style.use(latex_style_times)
    plt.figure()
    plt.plot(data[2, :, :nsamples].cpu(), results[:, :nsamples].cpu(), "k--")
    ax = plt.gca()
    
    nsamples = 100
    alpha = 0.05 / 2

    """
    times, strains, temps, cycles = experiments.make_tension_tests(
        torch.tensor([1.0e-2]), torch.tensor([0]), torch.tensor([0.5]), 200
    )
    data = torch.stack((times, temps, strains))
    control = torch.zeros(1, dtype=int)
    types = torch.zeros(1, dtype=int)
    """
    stress_results = torch.zeros(nsamples, data.shape[1])

    for i in trange(nsamples):
        stress_results[i, :] = sampler(data, cycles, types, control)[:, 0]

    mean_result = torch.mean(stress_results, 0)
    sresults, _ = torch.sort(stress_results, 0)
    min_result = sresults[int(alpha * nsamples), :]
    max_result = sresults[int((1 - alpha) * nsamples), :]

    
    (l,) = plt.plot(data[2, :, 0], mean_result, lw=4, color="k")
    p = plt.fill_between(data[2, :, 0], min_result, max_result, alpha=0.5, color="k")

    plt.legend(
        [
            Line2D([0], [0], color="k", ls="--"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="k", edgecolor=None, alpha=0.5),
        ],
        ["Experimental data", "Model average", "Model 95% prediction interval"],
        loc="best",
        prop={'size': 30}
    )

    plt.xlabel("Strain (mm/mm)", fontsize=30)
    plt.ylabel("Stress (MPa)", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.locator_params(axis="both", nbins=4)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("confidence.pdf")
    plt.show()
    plt.close()

