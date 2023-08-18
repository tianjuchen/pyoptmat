#!/usr/bin/env python3

import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib import RcParams
import warnings

latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)

def read_file(path, nsample):
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        batch = os.path.basename(f).split("-")[2].split(".csv")[0]
        if float(batch) == nsample:
            df = pd.read_csv(f)
            return df


def get_value(path, v, nsample=10):
    df = read_file(path, nsample)
    df2 = df[v].loc[(df[v] > 0.0) & (df[v] < 1.0)]
    return df2.values


def res(names, lists):
    for n, l in zip(names, lists):
        print("%s :" % (n), l)
    return names


def values(v, paths, batch=30):
    data = []
    for path in paths:
        data.append(get_value(path, v, nsample=batch))
    return data


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

def plot_score():
    df = pd.DataFrame(
        {
            "magnitude": np.array([613664.0, 643.0]),
            "varname": [r"$NUTS$", r"$SVI$"],
        }
    )
    sns.set(font_scale=1.5, style="white")
    plt.style.use(latex_style_times)
    p = sns.barplot(data=df, x="varname", y="magnitude", orient="v", log=True)
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 20})
    ax = plt.gca()
    plt.xlabel("", fontsize=18)
    plt.ylabel("Wall time (s)", fontsize=27)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.ylim(0, 10.5e5)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    show_values(p, space=0)
    plt.tight_layout()
    plt.savefig("nuts-vs-svi.pdf")
    plt.show()
    plt.close()
    return df


def tolist(res):
    data = []
    for i in res:
        if len(i) > 1:
            for j in i:
                data.append(j)
    # print(data)
    return data



if __name__ == "__main__":

    _ = plot_score()

    