#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import xarray as xr
import os.path
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import scipy.signal as sig
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import pyro.distributions as dist
from matplotlib import RcParams

if __name__ == "__main__":

    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]
    means = [0.51, 0.51, 0.50, 0.53, 0.50]
    stds = [0.08, 0.08, 0.08, 0.09, 0.09]

    amount = 10000

    latex_style_times = RcParams(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.usetex": True,
        }
    )

    for i, (mu, std, n, fn) in enumerate(zip(means, stds, rnames, names)):

        pdist = dist.Normal(mu, std).sample((amount,))
        tdist = dist.Normal(0.5, 0.15).sample((amount,))
        idist = dist.Normal(0.15, 0.15).sample((amount,))

        df = pd.DataFrame(
            {
                "dist": list(pdist.numpy()) + list(tdist.numpy()) + list(idist.numpy()),
                "catergorize": ["Posterior"] * amount
                + ["True"] * amount
                + ["Prior"] * amount,
            }
        )
        print(df)
        sns.set(font_scale=1.75, style="white")
        plt.style.use(latex_style_times)
        g = sns.kdeplot(
            data=df,
            x="dist",
            hue="catergorize",
            fill=True,
            common_norm=False,
            # palette="crest",
            # alpha=.5,
            linewidth=3,
            legend=True,
        )
        g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        plt.xlabel("{}".format(n), fontsize=30)
        plt.ylabel("Frequency", fontsize=30)
        plt.tick_params(axis="both", which="major", labelsize=30)
        plt.locator_params(axis="both", nbins=4)
        if i < len(names) - 1:
            plt.legend([], [], frameon=False)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("{}.pdf".format(fn))
        # plt.show()
        plt.close()
