#!/usr/bin/env python3

import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ra
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


def tolist(res):
    data = []
    for i in res:
        if len(i) > 1:
            for j in i:
                data.append(j)
    # print(data)
    return data



if __name__ == "__main__":

    param_comparison, time_comparison = True, False

    opt_lbfgs = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 524.21/3])
    opt_adam = np.array([0.50, 0.51, 0.49, 0.50, 0.49, 694.8975])
    point_2 = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 2129.90])
    # scipybgfs = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 2339.06])
    gradfree = np.array([0.50, 0.50, 0.45, 1.0, 0.0, 3367.29])

    optim_params = np.vstack((opt_lbfgs, opt_adam, point_2, gradfree))

    print(optim_params)


    real_names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    names = ["n", "eta", "s0", "R", "d"]
    methods = ["L-BFGS", "Adam", "Finite Difference", "Nelder-Mead"]

    values = []
    optim_methods = []
    input_params = []
    for i, n in enumerate(real_names):
        for j in range(optim_params.shape[0]):
            input_params.append(n)
            optim_methods.append(methods[j])
    for i in range(optim_params.shape[1]-1):
        for k in range(optim_params.shape[0]):
            values.append(optim_params[k, i])
            
            
    print(values)
    print(optim_methods)
    print(input_params)

    # save the stress-strain data for future use
    df = pd.DataFrame(
        {
            "values": values,
            "optim_methods": optim_methods,
            "input_params": input_params,
        }
    )

    print(df)
    # who v/s fare barplot
    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="input_params", y="values", hue="optim_methods")
    plt.legend(loc="best", frameon=False, ncol=1, prop={"size": 18})
    ax = plt.gca()
    plt.xlabel("Parameters", fontsize=18)
    plt.ylabel("Optimization Value", fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("optimize-optimizer-compare.pdf")
    plt.show()
    plt.close()


    # save the stress-strain data for future use
    df2 = pd.DataFrame(
        {
            "period": optim_params[:, -1],
            "methods": methods,
        }
    )
    print(df2)

    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df2, x="period", y="methods")
    plt.legend(loc="best", frameon=False, ncol=1, prop={"size": 18})
    ax = plt.gca()
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("", fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("optimizer-time-compare.pdf")
    plt.show()
    plt.close()

