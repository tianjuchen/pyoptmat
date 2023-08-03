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


def loc_scale_evolution(
    names, steps, means, scales, true_means, true_scales, Data_variability, fsize=23
):

    plt.style.use(latex_style_times)
    for name, mean in zip(names, means):

        plt.plot(steps, mean, label=name, ls="-", lw=3.0, marker="o", markersize=12)

    plt.plot(steps, true_means, label="True", ls="--", color="k", lw=3.0)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel(r"$\mu$", fontsize=fsize)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    plt.locator_params(axis="both", nbins=4)
    plt.legend(prop={"size": fsize}, frameon=False, ncol=2, loc="best")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("isotropic-mean-hist.pdf")
    plt.show()
    plt.close()

    for name, scale in zip(names, scales):

        plt.plot(steps, scale, label=name, ls="-", lw=3.0, marker="o", markersize=12)

    plt.plot(steps, true_scales, label="True", ls="--", color="k", lw=3.0)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel(r"$\sigma^{2}$", fontsize=fsize)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    plt.locator_params(axis="both", nbins=4)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("isotropic-variance-hist.pdf")
    plt.show()
    plt.close()
    return names


def iso_and_kin():
    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names += [r"$C_{1}$", r"$C_{2}$", r"$C_{3}$", r"$g_{1}$", r"$g_{2}$", r"$g_{3}$"]
    Data_variability = 0.15

    mean_1 = [0.94, 0.04, 0.54, 0.88, 0.57]
    mean_1 += [0.76, 0.17, 0.11, 0.34, 0.40, 0.13]

    mean_2 = [0.4844, 0.4896, 0.4940, 0.5096, 0.5094]
    mean_2 += [0.4843, 0.4791, 0.4591, 0.4641, 0.5400, 0.5620]

    mean_3 = [0.4792, 0.4916, 0.4827, 0.5096, 0.5014]
    mean_3 += [0.5109, 0.5055, 0.4592, 0.5152, 0.5770, 0.5869]

    mean_4 = [0.4787, 0.4920, 0.4816, 0.5115, 0.5000]
    mean_4 += [0.5110, 0.5030, 0.4646, 0.5161, 0.5716, 0.5616]

    mean_5 = [0.4792, 0.4928, 0.4824, 0.5101, 0.4982]
    mean_5 += [0.5114, 0.5036, 0.4676, 0.5167, 0.5740, 0.5588]

    mean_6 = [0.48, 0.49, 0.48, 0.51, 0.50]
    mean_6 += [0.51, 0.50, 0.47, 0.52, 0.57, 0.56]

    scale_1 = [0.15] * len(names)

    scale_2 = [0.1368, 0.1187, 0.0988, 0.1357, 0.146]
    scale_2 += [0.1185, 0.1161, 0.1320, 0.2581, 0.2205, 0.1758]

    scale_3 = [0.1104, 0.1331, 0.1088, 0.13, 0.14]
    scale_3 += [0.1439, 0.1477, 0.1913, 0.1951, 0.1985, 0.3217]

    scale_4 = [0.1121, 0.1320, 0.1095, 0.127, 0.138]
    scale_4 += [0.1429, 0.1503, 0.1916, 0.1947, 0.1986, 0.3389]

    scale_5 = [0.1130, 0.1317, 0.1104, 0.123, 0.132]
    scale_5 += [0.1439, 0.1512, 0.1915, 0.1944, 0.2006, 0.3389]

    scale_6 = [0.11, 0.13, 0.11, 0.1257, 0.136]
    scale_6 += [0.14, 0.15, 0.19, 0.19, 0.20, 0.35]

    steps = np.linspace(0.0, 500.0, 6)

    true_means = [0.50] * len(steps)
    true_scales = [Data_variability] * len(steps)

    means = list(np.vstack((mean_1, mean_2, mean_3, mean_4, mean_5, mean_6)).T)
    scales = list(np.vstack((scale_1, scale_2, scale_3, scale_4, scale_5, scale_6)).T)

    loc_scale_evolution(
        names[:5],
        steps,
        means[:5],
        scales[:5],
        true_means,
        true_scales,
        Data_variability,
    )
    return names


if __name__ == "__main__":

    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    Data_variability = 0.15

    mean_1 = [0.63, 0.54, 0.80, 0.19, 0.62]
    mean_2 = [0.5166, 0.4338, 0.6865, 0.0837, 0.5066]
    mean_3 = [0.3166, 0.2455, 0.4865, 0.1716, 0.3066]
    mean_4 = [0.4639, 0.4659, 0.4720, 0.4215, 0.4435]
    mean_5 = [0.4785, 0.4785, 0.4787, 0.4227, 0.4179]
    mean_6 = [0.51, 0.50, 0.51, 0.52, 0.53]

    scale_1 = [0.15] * len(names)
    scale_2 = [0.1674, 0.1674, 0.1674, 0.1478, 0.1674]
    scale_3 = [0.1911, 0.1811, 0.2041, 0.1253, 0.1900]
    scale_4 = [0.1010, 0.0936, 0.1187, 0.1211, 0.1006]
    scale_5 = [0.1049, 0.1047, 0.1057, 0.1387, 0.1198]
    scale_6 = [0.10, 0.10, 0.10, 0.13, 0.14]

    steps = np.linspace(0.0, 500.0, 6)

    true_means = [0.50] * len(steps)
    true_scales = [Data_variability] * len(steps)

    means = list(np.vstack((mean_1, mean_2, mean_3, mean_4, mean_5, mean_6)).T)
    scales = list(np.vstack((scale_1, scale_2, scale_3, scale_4, scale_5, scale_6)).T)

    loc_scale_evolution(
        names,
        steps,
        means,
        scales,
        true_means,
        true_scales,
        Data_variability,
    )
