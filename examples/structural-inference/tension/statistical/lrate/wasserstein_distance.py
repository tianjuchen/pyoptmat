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
import torch.nn.functional as F
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wasserstein_distance

def read_file(path, file_name):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        ffname = os.path.basename(f).split(".txt")[0]
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


def KL_div(mu, std, true_mu=0.5, true_std=0.15, amount=5000000):
    pdist = torch.distributions.normal.Normal(mu, std).sample((amount,))
    tdist = torch.distributions.normal.Normal(true_mu, true_std).sample((amount,))

    p = F.softmax(pdist, dim=-1)
    t = F.softmax(tdist, dim=-1)

    return F.kl_div(p, t, reduction="sum").abs().item()



def bhatta_loss(mu, std, true_mu=0.5, true_std=0.15, amount=1000):
    pdist = torch.distributions.normal.Normal(mu, std).sample((amount,))
    tdist = torch.distributions.normal.Normal(true_mu, true_std).sample((amount,))

    p = F.softmax(pdist, dim=-1)
    t = F.softmax(tdist, dim=-1)

    out = -torch.log(torch.sum(torch.sqrt(torch.abs(torch.mul(t, t)))))
    return out.item()


def z_test(mu, std, true_mu=0.5, true_std=0.15, amount=1000):
    pdist = torch.distributions.normal.Normal(mu, std).sample((amount,))
    tdist = torch.distributions.normal.Normal(true_mu, true_std).sample((amount,))

    pm, ms = torch.std_mean(pdist)
    tm, ts = torch.std_mean(tdist)

    out = (pm-tm)/torch.sqrt(ms**2+ts**2)
    return out.abs().item()
    
def wasserstein(mu, std, true_mu=0.5, true_std=0.15, amount=1000):
    pdist = torch.distributions.normal.Normal(mu, std).sample((amount,))
    tdist = torch.distributions.normal.Normal(true_mu, true_std).sample((amount,))
    out = wasserstein_distance(pdist.numpy(), tdist.numpy())
    return out


def kl_param():

    rnames = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]

    lrates = [r"$\gamma=0.1$", r"$\gamma=0.01$", r"$\gamma=0.001$"]

    lrate1_mean = [0.51, 0.49, 0.52, 0.53, 0.50]
    lrate1_std = [0.06, 0.06, 0.06, 0.12, 0.12]

    lrate2_mean = [0.51, 0.51, 0.50, 0.53, 0.50]
    lrate2_std = [0.08, 0.08, 0.08, 0.09, 0.09]

    lrate3_mean = [0.34, 0.34, 0.34, 0.34, 0.34]
    lrate3_std = [0.40, 0.40, 0.40, 0.40, 0.40]

    lrate_mean = np.vstack((lrate1_mean, lrate2_mean, lrate3_mean))
    lrate_std = np.vstack((lrate1_std, lrate2_std, lrate3_std))

    kl_score = []
    for i in range(lrate_mean.shape[1]):
        for j in range(lrate_mean.shape[0]):
            print(lrate_mean[j, i], lrate_std[j, i])
            score = wasserstein(lrate_mean[j, i], lrate_std[j, i])# * 1.0e6
            kl_score.append(score)

    # normalize kl divergence
    # kl_score = np.array(kl_score) / max(kl_score)

    params = []
    for i in rnames:
        params += [i] * 3

    df = pd.DataFrame({"kl": kl_score, "params": params, "lrate": lrates * 5})

    sns.set(font_scale=1.5, style="white")
    plt.style.use(latex_style_times)
    plt.figure(figsize=(6.4, 4.8))
    sns.barplot(data=df, x="params", y="kl", hue="lrate")# palette='Greys', edgecolor='k')
    plt.legend(loc="best", frameon=False, ncol=1, prop={"size": 18})
    
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.yscale('log')  # set log-scale y axis
    ax = plt.gca()    
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_minor_formatter(NullFormatter())
    
    # ax.set_yticks([15.4135, 15.4137, 15.4139])
    # ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    ax.set_xlabel("", fontsize=23)
    ax.set_ylabel("Wasserstein Distance", fontsize=23)
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("Wasserstein-distance-lrate.pdf")
    plt.show()
    plt.close()
    return names


def loss_plot():
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/statistical/lrate/"
    bpath = path1 + path2

    fpath = ["lrate-1/", "lrate-2/", "lrate-3/"]

    fnames = [
        "loss-history-lrate-0.1",
        "loss-history-lrate-0.01",
        "loss-history-lrate-0.001",
    ]

    lrs = [0.1, 0.01, 0.001]

    fsize = 30

    plt.figure(figsize=(6.4, 4.8))
    for fn, fp, lr in zip(fnames, fpath, lrs):
        df = read_file(bpath + fp, fn)
        plt.style.use(latex_style_times)
        plt.plot(df, lw=3, label=r"$\gamma={}$".format(lr))
    
    ax = plt.gca()
    plt.xlabel("Step", fontsize=fsize)
    plt.ylabel("ELBO", fontsize=fsize)
    plt.tick_params(axis="both", which="major", labelsize=fsize)
    ax.locator_params(nbins=4, axis='both')
    plt.legend(frameon=False, prop={"size": 20})
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)

    plt.tight_layout()
    plt.savefig("loss-lrate.pdf")
    plt.show()
    plt.close()
    return fsize


if __name__ == "__main__":

    # _ = kl_param()

    _ = loss_plot()