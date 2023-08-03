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
from matplotlib.ticker import FormatStrFormatter


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


if __name__ == "__main__":
    amount = 100

    loc = torch.tensor([0.4350, 0.9901, 0.2395, 0.1107, 0.9076])
    scale = torch.tensor([0.0280, 0.0265, 0.0250, 0.0261, 0.0265])
    scale_tril = torch.tensor(
        [
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [-0.0055, 1.0000, 0.0000, 0.0000, 0.0000],
            [0.0175, 0.0793, 1.0000, 0.0000, 0.0000],
            [-0.0767, -0.0497, -0.0377, 1.0000, 0.0000],
            [0.0870, -0.1439, 0.2029, -0.1275, 1.0000],
        ],
    )

    pdist = dist.Normal(loc, scale).sample((amount,))

    data = pd.DataFrame(
        {
            "n": pdist[:, 0].numpy(),
            "eta": pdist[:, 1].numpy(),
            "s0": pdist[:, 2].numpy(),
            "R": pdist[:, 3].numpy(),
            "d": pdist[:, 4].numpy(),
        }
    )

    sns.set(font_scale=1.5, style="white")
    plt.style.use(latex_style_times)
    sns.pairplot(data)
    ax = plt.gca()
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.locator_params(axis="both", nbins=4)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("pair-plot.pdf")
    # plt.show()
    plt.close()    

    """
    for i, (m, s) in enumerate(zip(means, stds)):
        param = []
        cT = []
        name = []
        for j, (T, cm, cs) in enumerate(zip(Ts, m, s)):
            pdist = dist.Normal(cm, cs).sample((amount,))
            param += list(pdist.numpy())
            cT += [T.item() for k in range(amount)]
            name += [names[i]] * amount

        data = pd.DataFrame(
            {
                "param": param,
                "name": name,
                "temperature": cT
            }
        )
        print(data)
        sns.set(font_scale=1.5, style="white")
        plt.style.use(latex_style_times)
        g = sns.histplot(
            data=data,
            y="param",
            x="temperature",
            bins=10,
            discrete=(False, False),
            cbar=True, 
            cbar_kws=dict(shrink=.75),
            legend=False
        )
        # g.legend_.set_title(None)
        ax = plt.gca()
        plt.xlabel("Temperature ($^\circ$C)", fontsize=30)
        plt.ylabel("{}", fontsize=30, rotation=90)
        # if i == 0:
            # plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        # else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        plt.tick_params(axis="both", which="major", labelsize=30)
        plt.locator_params(axis="both", nbins=4)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("a800h-{}.pdf".format(fnames[i]))
        # plt.show()
        plt.close()
        
        # sys.exit("stop")
    """
