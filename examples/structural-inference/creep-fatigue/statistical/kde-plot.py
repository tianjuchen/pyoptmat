#!/usr/bin/env python3
import sys
import os, glob
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


def read_file(path, file_name):
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        ffname = os.path.basename(f).split(".csv")[0]
        if ffname == file_name:
            df = pd.read_csv(f)
            return df


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)


def calavg(mus):
    avg_mu = []
    for mu in mus:
        v = np.mean([df[mu].to_numpy() for df in dfs])
        avg_mu.append(v)
    return avg_mu


if __name__ == "__main__":

    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    rnames += [r"$C_{1}$", r"$C_{2}$", r"$C_{3}$", r"$g_{1}$", r"$g_{2}$", r"$g_{3}$"]

    names = ["n", "eta", "s0", "R", "d"]
    names += ["C1", "C2", "C3", "g1", "g2", "g3"]

    avg_mu = [0.48, 0.49, 0.49, 0.52, 0.51, 0.51, 0.51, 0.47, 0.51, 0.58, 0.54]
    avg_std = [0.11, 0.13, 0.12, 0.11, 0.10, 0.14, 0.14, 0.19, 0.21, 0.22, 0.34]

    amount = 100000
    for i, (mu, std, n, fn) in enumerate(zip(avg_mu, avg_std, rnames, names)):

        pdist = dist.Normal(mu, std).sample((amount,))
        tdist = dist.Normal(0.5, 0.15).sample((amount,))
        data = pd.DataFrame(
            {
                "dist": list(pdist.numpy()) + list(tdist.numpy()),
                "catergorize": ["Posterior"] * amount + ["True"] * amount,
            }
        )

        sns.set(font_scale=1.75, style="white")
        plt.style.use(latex_style_times)
        g = sns.kdeplot(
            data=data,
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
