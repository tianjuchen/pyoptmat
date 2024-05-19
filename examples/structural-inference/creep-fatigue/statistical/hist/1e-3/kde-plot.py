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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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


if __name__ == "__main__":
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/statistical//hist/1e-3/"

    path = path1 + path2

    # _ = dist_box(path, amount=50000, fs=25)

    fns = "loss-history-hist-"
    scales = ["0.05", "0.1", "0.15"]

    for scale in scales:
        df = read_file(path, fns + scale, ftype=".txt")
        plt.style.use(latex_style_times)
        plt.plot(df, lw=3)  # , label=r"$\sigma^{2}$" + "={}".format(float(scale)))
    plt.yscale("log")
    fsize = 30
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel("ELBO", fontsize=fsize)
    # plt.ylim(bottom=None, top=10**18)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    ax.locator_params(nbins=4, axis="x")
    plt.legend(
        [
            Line2D([0], [0], color="#1f77b4", lw=4),
            Line2D([0], [0], color="#ff7f0e", lw=4),
            Line2D([0], [0], color="#2ca02c", lw=4),
        ],
        [
            r"$\sigma_{sample}^{2}$" + "= 0.05",
            r"$\sigma_{sample}^{2}$" + "= 0.1",
            r"$\sigma_{sample}^{2}$" + "= 0.15",
        ],
        loc="best",
        prop={"size": 17},
    )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("cyclic-random-loss.pdf")
    plt.show()
    plt.close()