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


def convert_string_to_data(df):
    data = []
    for i in df:
        data.append(eval(i)[0])
    return np.array(data).mean()


def random_posterior():
    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/repeat/"

    fpath = ["random-5/", "random-10/", "random-15/"]

    mus = ["n_mu", "eta_mu", "s0_mu", "R_mu", "d_mu"]
    stds = ["n_std", "eta_std", "s0_std", "R_std", "d_std"]

    fs = ["res-0.05-30", "res-0.10-30", "res-0.15-30"]

    df = read_file(path1 + path2 + fpath[0], fs[0])

    amount = 10000
    scale = 0.05
    p = []
    for m, s in zip(mus, stds):
        mu = convert_string_to_data(df[m])
        std = convert_string_to_data(df[s])
        pdist = dist.Normal(mu, std).sample((amount,))
        p += list(pdist.numpy())
    tdist = dist.Normal(0.5, scale).sample((amount,))

    params = []
    for n in rnames:
        params += [n] * amount

    data = pd.DataFrame(
        {
            "dist": p + list(tdist.numpy()),
            "params": params + ["True"] * amount,
        }
    )
    print(data)
    sns.set(font_scale=1.5, style="white")
    plt.style.use(latex_style_times)
    g = sns.boxplot(
        data=data,
        y="dist",
        x="params",
    )
    # g.legend_.set_title(None)
    ax = plt.gca()
    plt.xlabel("{}", fontsize=30)
    plt.ylabel("{}", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    # plt.locator_params(axis="both", nbins=3)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("random-{}.pdf".format(scale))
    plt.show()
    plt.close()
    return df


def plt_loss():
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/repeat/"

    fpath = ["random-5/", "random-10/", "random-15/"]

    fs = ["loss-history-0.05-0", "loss-history-0.1-0", "loss-history-0.15-0"]
    scale = [0.05, 0.1, 0.15]

    for i in range(len(fs)):
        df = read_file(path1 + path2 + fpath[i], fs[i])

        plt.style.use(latex_style_times)
        plt.plot(df, lw=4, label=r"$\sigma^{2}$" + "={}".format(scale[i]))

    plt.yscale("log")
    ax = plt.gca()
    plt.xlabel("Step", fontsize=30)
    plt.ylabel(r"$ELBO$", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.legend(loc="upper right", frameon=False, ncol=1, prop={"size": 25})
    plt.locator_params(axis="x", nbins=3)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("random-loss.pdf")
    plt.show()
    plt.close()
    return scale


if __name__ == "__main__":
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/hist/"

    fs = ["hist-0.05-30", "hist-0.10-30", "hist-0.15-30"]
    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    mus = ["n_mu", "eta_mu", "s0_mu", "R_mu", "d_mu"]
    stds = ["n_std", "eta_std", "s0_std", "R_std", "d_std"]

    df = read_file(path1 + path2, fs[2])

    for i in range(len(rnames)):
        plt.style.use(latex_style_times)
        plt.plot(df[stds[i]], "o-", lw=4, markersize=15, markevery=50, label=rnames[i])

    plt.plot(np.array([0.15] * 300), "k--", lw=4, label="True")
    # plt.xlim(-1, 300)
    # plt.ylim(0, 1)
    ax = plt.gca()
    plt.xlabel("Step", fontsize=30)
    plt.ylabel("{}", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    # plt.legend(loc="lower right", frameon=False, ncol=2, prop={"size": 25})
    plt.locator_params(axis="both", nbins=2)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("variance-hist-{}.pdf".format(0.15))
    plt.show()
    plt.close()
