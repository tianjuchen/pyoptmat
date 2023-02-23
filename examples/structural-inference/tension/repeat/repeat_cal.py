#!/usr/bin/env python3

import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns


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
    return df2.mean()


def res(names, lists):
    for n, l in zip(names, lists):
        print("%s :" % (n), l)
    return names


def values(paths, batch=30):
    ns = []
    etas = []
    s0s = []
    Rs = []
    ds = []
    for path in paths:
        ns.append(get_value(path, "n", nsample=batch))
        etas.append(get_value(path, "eta", nsample=batch))
        s0s.append(get_value(path, "s0", nsample=batch))
        Rs.append(get_value(path, "R", nsample=batch))
        ds.append(get_value(path, "d", nsample=batch))

    lists = [
        np.array(ns).mean(),
        np.array(etas).mean(),
        np.array(s0s).mean(),
        np.array(Rs).mean(),
        np.array(ds).mean(),
    ]

    if batch == 10:
        bn = "Ten"
    elif batch == 20:
        bn = "Twenty"
    elif batch == 30:
        bn = "Thirty"
    return np.array(lists + [batch])


def plot_score():
    df = pd.DataFrame(
        {
            "magnitude": np.array([2.06, 2.05, 2.44]),
            "varname": np.array([10, 20, 30]),
        }
    )
    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="varname", y="magnitude")
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 20})
    ax = plt.gca()
    plt.xlabel(r"$n_{sample}$", fontsize=18)
    plt.ylabel("Strict Accuracy Score (%)", fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    # plt.savefig("batch-score-compare.pdf")
    plt.show()
    plt.close()
    return df


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/repeat/"
    bspath = path1 + path2

    path_b1 = bspath + "repeat/"
    path_b2 = bspath + "repeat2/"
    path_b3 = bspath + "repeat3/"
    path_b4 = bspath + "repeat4/"
    path_b5 = bspath + "repeat5/"

    paths = [path_b1, path_b2, path_b3, path_b4, path_b5]

    result_10 = values(paths, batch=10)
    result_20 = values(paths, batch=20)
    result_30 = values(paths, batch=30)

    data_list = np.vstack((result_10, result_20, result_30))

    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]

    opt_values = []
    opt_vars = []
    opt_batchs = []
    for i, n in enumerate(names):
        opt_values.append(data_list[:, i])
        opt_vars.append([n for i in range(data_list.shape[0])])
        opt_batchs.append(data_list[:, -1])

    print(data_list)
    valuess = np.abs(np.array(opt_values).flatten() - 0.5) / 0.5 * 100.0
    varss = np.array(opt_vars).flatten()
    batchss = np.array(opt_batchs).flatten()

    print(valuess)
    print(varss)
    print(batchss)

    # save the stress-strain data for future use
    df = pd.DataFrame(
        {
            "magnitude": valuess,
            "varname": varss,
            "batch": batchss,
        }
    )

    print(df)
    # who v/s fare barplot
    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="batch", y="magnitude", hue="varname")
    plt.legend(loc="upper right", frameon=False, ncol=2, prop={"size": 20})
    ax = plt.gca()
    plt.xlabel(r"$n_{sample}$", fontsize=18)
    plt.ylabel("Optimization gap (%)", fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    # plt.savefig("batch-gap-compare.pdf")
    plt.show()
    plt.close()

    _ = plot_score()
