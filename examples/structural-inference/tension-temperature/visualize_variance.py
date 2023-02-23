#!/usr/bin/env python3

"""
    Simple helper to make a plot illustrating the variation in the
    synthetic experimental data.
"""

import sys

sys.path.append("../../..")

import xarray as xr
import torch
import matplotlib.pyplot as plt
from maker import make_model, downsample
from pyoptmat import optimize, experiments

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    for scale in scales:
        input_data = xr.load_dataset("scale-%3.2f.nc" % scale)
        data, results, cycles, types, control = downsample(
                experiments.load_results(input_data),
                50,
                input_data.nrates,
                input_data.nsamples,
            )
        
        print(data[1])
        strain = data[2]
        stress = results
        # strain = data.strain.data.reshape(-1, data.nrates, data.nsamples)
        # stress = data.stress.data.reshape(-1, data.nrates, data.nsamples)

        ax = plt.gca()
        # plt.plot(strain[:, 0], stress[:, 0], lw=3)
        plt.plot(strain.numpy(), stress.numpy(), lw=3)
        plt.xlabel("Strain (mm/mm)", fontsize=27)
        plt.ylabel("Stress (MPa)", fontsize=27)
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=27)
        plt.locator_params(axis='both', nbins=3)
        plt.tick_params(axis="both", which="major", labelsize=27)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        # plt.title("Scale = %3.2f" % scale)
        plt.tight_layout()
        plt.savefig("tension-noise-visualize-%3.2f.pdf" % scale)
        # plt.show()
        plt.close()
