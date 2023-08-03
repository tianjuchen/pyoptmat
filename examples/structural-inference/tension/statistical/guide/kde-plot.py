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


def read_file(path, file_name, ftype=".csv"):
    fnames = glob.glob(path + "*" + ftype)
    for f in fnames:
        ffname = os.path.basename(f).split(ftype)[0]
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


def z_test(mu, std, true_mu=0.5, true_std=0.15, amount=5000000):
    pdist = torch.distributions.normal.Normal(mu, std).sample((amount,))
    tdist = torch.distributions.normal.Normal(true_mu, true_std).sample((amount,))

    pm, ms = torch.std_mean(pdist)
    tm, ts = torch.std_mean(tdist)

    out = (pm - tm) / torch.sqrt(ms**2 + ts**2)
    return out.abs().item()


def calavg(mus):
    avg_mu = []
    for mu in mus:
        v = np.mean([df[mu].to_numpy() for df in dfs])
        avg_mu.append(v)
    return avg_mu


def dist_box(path, amount=1000, fs=27, fns="res-30-"):
    scales = ["0.05", "0.1", "0.15"]
    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    rnames = [
        r"$n$",
        r"$\eta$",
        r"$s_{0}$",
        r"$R$",
        r"$d$",
        r"$C_{1}$",
        r"$C_{2}$",
        r"$C_{3}$",
        r"$g_{1}$",
        r"$g_{2}$",
        r"$g_{3}$",
    ]

    for scale in scales:
        df = read_file(path, fns + scale)
        catergory = ["mu", "std"]
        p = []
        for n in names:
            vnm = n + "_" + catergory[0]
            vns = n + "_" + catergory[1]
            pm = df[vnm].mean()
            ps = df[vns].mean()
            pdist = dist.Normal(pm, ps).sample((amount,))
            p += list(pdist.numpy())

        tdist = dist.Normal(0.5, float(scale)).sample((amount,))

        params = []
        for n in rnames:
            params += [n] * amount

        data = pd.DataFrame(
            {
                "dist": p + list(tdist.numpy()),
                "params": params + ["True"] * amount,
            }
        )

        sns.set(font_scale=1.75, style="white")
        plt.style.use(latex_style_times)
        plt.figure(figsize=(6.4, 4.8))
        g = sns.boxplot(
            data=data,
            x="dist",
            y="params",
            # palette='Greys'
        )
        # g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlabel("{}", fontsize=fs)
        plt.ylabel("{}", fontsize=fs)
        plt.tick_params(axis="both", which="major", labelsize=fs)
        plt.locator_params(axis="x", nbins=4)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        # plt.savefig(path + "cyclic-posterior-{}.pdf".format(float(scale)))
        plt.show()
        plt.close()
    return names


def plot_loss():
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/guide/"

    path = path1 + path2

    fns = "loss-history-"
    scales = ["Beta", "Normal", "MAP"]

    for scale in scales:
        df = read_file(path, fns + scale, ftype=".txt")
        plt.style.use(latex_style_times)
        plt.plot(df[:300], lw=3, label="{}".format(scale))
    plt.yscale("log")
    fsize = 30
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel("ELBO", fontsize=fsize)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    ax.locator_params(nbins=4, axis="x")
    plt.legend(frameon=False, prop={"size": 25})
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig(path + "guide-random-loss.pdf")
    # plt.show()
    plt.close()
    return scales


def plot_guide(fs=30, amount = 50000):

    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    rnames = [
        r"$n$",
        r"$\eta$",
        r"$s_{0}$",
        r"$R$",
        r"$d$",
        r"$C_{1}$",
        r"$C_{2}$",
        r"$C_{3}$",
        r"$g_{1}$",
        r"$g_{2}$",
        r"$g_{3}$",
    ]

    mu_beta = [0.5674, 0.9205, 0.5826, 0.5625, 0.4243]
    std_beta = [0.3106, 0.2081, 0.1940, 0.2804, 0.1962]

    mu_n = [0.4205, 0.4030, 0.5066, 0.4931, 0.5060]
    std_n = [0.2576, 0.3518, 0.2073, 0.2527, 0.2878]

    mu_m = [0.50, 0.50, 0.50, 0.47, 0.50]
    std_m = [0.09, 0.09, 0.09, 0.15, 0.15]

    mus = np.vstack((mu_beta, mu_n, mu_m))
    stds = np.vstack((std_beta, std_n, std_m))

    guides = ["Beta", "Normal", "MAP"]
    
    p = []
    for i in range(mus.shape[-1]):
        bdist = dist.Beta(mus[0, i], stds[0, i]).sample((amount,))
        ndist = dist.Normal(mus[1, i], stds[1, i]).sample((amount,))
        pdist = dist.Normal(mus[2, i], stds[2, i]).sample((amount,))
        tdist = dist.Normal(0.5, 0.15).sample((amount,))

        guide = []
        for g in guides:
            guide += [g] * amount

        data = pd.DataFrame(
            {
                "dist": list(bdist.numpy())
                + list(ndist.numpy())
                + list(pdist.numpy())
                + list(tdist.numpy()),
                "guide": guide + ["True"] * amount,
            }
        )

        sns.set(font_scale=1.75, style="white")
        plt.style.use(latex_style_times)
        plt.figure(figsize=(6.4, 4.8))
        g = sns.boxplot(
            data=data,
            x="dist",
            y="guide",
            # palette='Greys'
        )
        # g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlabel("{}", fontsize=fs)
        plt.ylabel("{}", fontsize=fs)
        plt.tick_params(axis="both", which="major", labelsize=fs)
        plt.locator_params(axis="x", nbins=2)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("guide-posterior-{}.pdf".format(names[i]))
        # plt.show()
        plt.close()


if __name__ == "__main__":

    _ = plot_loss()