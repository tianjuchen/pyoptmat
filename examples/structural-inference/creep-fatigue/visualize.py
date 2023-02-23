import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":
    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    erange = 0.02
    hold_time = 5 * 60.0

    for scale in scales:
        database = xr.open_dataset("scale-%3.2f.nc" % scale)
        use = database.where(
            np.logical_and(
                np.abs(database.strain_ranges - erange) <= 1e-6,
                np.abs(database.hold_times - hold_time) <= 1e-6,
            ),
            drop=True,
        )
        ax = plt.gca()
        # plt.figure(figsize=(6.4, 4.8))
        plt.plot(use.strain[:, 0, :], use.stress[:, 0, :], lw=3)
        plt.xlabel("Strain (mm/mm)", fontsize=27)
        plt.ylabel("Stress (MPa)", fontsize=27)
        # plt.xticks(fontsize=27)
        # plt.yticks(fontsize=27)
        plt.locator_params(axis='both', nbins=3)
        plt.tick_params(axis="both", which="major", labelsize=27)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig("cyclic-visualize-%3.2f.pdf" % scale)
        plt.show()
        plt.close()
