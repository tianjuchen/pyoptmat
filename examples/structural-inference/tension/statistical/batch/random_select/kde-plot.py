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


def dist_comparison():
    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]

    batch = [5, 10, 20, 30]
    mu_b5 = [0.52, 0.52, 0.52, 0.55, 0.46]
    std_b5 = [0.06, 0.06, 0.06, 0.10, 0.11]

    mu_b10 = [0.55, 0.55, 0.55, 0.54, 0.56]
    std_b10 = [0.08, 0.09, 0.08, 0.15, 0.07]

    mu_b20 = [0.50, 0.50, 0.50, 0.48, 0.49]
    std_b20 = [0.07, 0.07, 0.07, 0.13, 0.09]

    mu_b30 = [0.49, 0.49, 0.49, 0.50, 0.54]
    std_b30 = [0.08, 0.08, 0.08, 0.11, 0.12]

    mu_bs = np.vstack((mu_b5, mu_b10, mu_b20, mu_b30))
    std_bs = np.vstack((std_b5, std_b10, std_b20, std_b30))

    amount = 500
    colors = ["Reds", "Blues", "Greens", "Purples"]

    for j in range(mu_bs.shape[-1]):
        p = []
        for i in range(mu_bs.shape[0]):
            pdist = dist.Normal(mu_bs[i, j], std_bs[i, j]).sample((amount,))
            p += list(pdist.numpy())

        tdist = dist.Normal(mu_bs[i, j], std_bs[i, j]).sample((amount,))

        data = pd.DataFrame(
            {
                "dist": p + list(tdist.numpy()),
                "param": [r"$n_{sample} = 5$"] * amount
                + [r"$n_{sample} = 10$"] * amount
                + [r"$n_{sample} = 20$"] * amount
                + [r"$n_{sample} = 30$"] * amount
                + ["True"] * amount,
            }
        )
        print(data)
        sns.set(font_scale=1.5, style="white")
        plt.style.use(latex_style_times)
        g = sns.kdeplot(
            data=data,
            x="dist",
            hue="param",
            fill=True,
            common_norm=False,
            # palette="crest",
            # alpha=.5,
            shade=False,
            linewidth=7,
            legend=True,
        )
        g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        plt.xlabel("{}".format(rnames[j]), fontsize=30)
        plt.ylabel("Density", fontsize=30)
        plt.tick_params(axis="both", which="major", labelsize=30)
        plt.locator_params(axis="both", nbins=3)
        if j > 0:
            plt.legend([], frameon=False)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("{}.pdf".format(names[j]))
        plt.show()
        plt.close()
    return names


def dist_score():
    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]

    batch = [5, 10, 20, 30]
    mu_b5 = [0.52, 0.52, 0.52, 0.55, 0.46]
    std_b5 = [0.06, 0.06, 0.06, 0.10, 0.11]

    mu_b10 = [0.55, 0.55, 0.55, 0.54, 0.56]
    std_b10 = [0.08, 0.09, 0.08, 0.15, 0.07]

    mu_b20 = [0.50, 0.50, 0.50, 0.48, 0.49]
    std_b20 = [0.07, 0.07, 0.07, 0.13, 0.09]

    mu_b30 = [0.49, 0.49, 0.49, 0.50, 0.54]
    std_b30 = [0.08, 0.08, 0.08, 0.11, 0.12]

    mu_bs = np.vstack((mu_b5, mu_b10, mu_b20, mu_b30))
    std_bs = np.vstack((std_b5, std_b10, std_b20, std_b30))

    amount = 500
    colors = ["Reds", "Blues", "Greens", "Purples"]

    p = []
    for j in range(mu_bs.shape[-1]):
        for i in range(mu_bs.shape[0]):
            score = z_test(mu_bs[i, j], std_bs[i, j], amount=50000000)
            p.append(score)

    params = []
    for n in rnames:
        params += [n] * len(batch)

    df = pd.DataFrame(
        {
            "zscore": p,
            "nsample": [
                r"$n_{sample} = 5$",
                r"$n_{sample} = 10$",
                r"$n_{sample} = 20$",
                r"$n_{sample} = 30$",
            ]
            * len(names),
            "params": params,
        }
    )

    sns.set(font_scale=1.5, style="white")
    plt.style.use(latex_style_times)
    plt.figure(figsize=(6.4, 4.8))
    sns.barplot(
        data=df, x="params", y="zscore", hue="nsample"
    )  # palette='Greys', edgecolor='k')
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 18})

    plt.ylim(0, 0.175)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax = plt.gca()
    ax.set_xlabel("", fontsize=23)
    ax.set_ylabel("Divergence", fontsize=23)
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("ztest-divergence-sample.pdf")
    plt.show()
    plt.close()
    return names


def dist_box():
    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]

    batch = [5, 10, 20, 30]
    mu_b5 = [0.52, 0.52, 0.52, 0.55, 0.46]
    std_b5 = [0.06, 0.06, 0.06, 0.10, 0.11]

    mu_b10 = [0.55, 0.55, 0.55, 0.54, 0.56]
    std_b10 = [0.08, 0.09, 0.08, 0.15, 0.07]

    mu_b20 = [0.50, 0.50, 0.50, 0.48, 0.49]
    std_b20 = [0.07, 0.07, 0.07, 0.13, 0.09]

    mu_b30 = [0.49, 0.49, 0.49, 0.50, 0.54]
    std_b30 = [0.08, 0.08, 0.08, 0.11, 0.12]

    mu_bs = np.vstack((mu_b5, mu_b10, mu_b20, mu_b30))
    std_bs = np.vstack((std_b5, std_b10, std_b20, std_b30))

    amount = 50000
    colors = ["Reds", "Blues", "Greens", "Purples"]

    for j in range(mu_bs.shape[-1]):
        p = []
        for i in range(mu_bs.shape[0]):
            pdist = dist.Normal(mu_bs[i, j], std_bs[i, j]).sample((amount,))
            p += list(pdist.numpy())
        tdist = dist.Normal(0.5, 0.15).sample((amount,))

        nsample = (
            [5] * amount
            + [10] * amount
            + [20] * amount
            + [30] * amount
        )

        params = []
        for n in rnames:
            params += [n] * len(batch) * amount

        data = pd.DataFrame(
            {
                "dist": p + list(tdist.numpy()),
                "nsample": nsample + ["True"] * amount,
            }
        )
        print(data)
        sns.set(font_scale=1.5, style="white")
        plt.style.use(latex_style_times)
        g = sns.boxplot(
            data=data,
            y="dist",
            x="nsample",
            # palette='Greys'
        )
        # g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlabel(r"$n_{sample}$", fontsize=30)
        plt.ylabel("{}", fontsize=30)
        plt.tick_params(axis="both", which="major", labelsize=30)
        # plt.locator_params(axis="both", nbins=3)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("box-{}.pdf".format(names[j]))
        # plt.show()
        plt.close()
    return names


if __name__ == "__main__":

    _ = dist_box()
