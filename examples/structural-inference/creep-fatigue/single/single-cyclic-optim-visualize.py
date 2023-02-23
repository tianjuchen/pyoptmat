#!/usr/bin/env python3
import sys
import os.path
import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def optim(name, params, lrs, fs=18):

    marker_shapes = ["o", "^", "*"]

    for param, lr, marker_shape in zip(params, lrs, marker_shapes):
        plt.plot(param, lr, marker=marker_shape, markersize=fs)
        plt.axvline(x=0.5, color="k", lw=3, linestyle="--")

    ax = plt.gca()
    plt.yscale("log")
    plt.xlim([0, 1])
    plt.xlabel("{}".format(name), fontsize=fs)
    plt.ylabel("Learning rate", fontsize=fs)
    # plt.locator_params(axis='both', nbins=4)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    # plt.title("Scattering Predictive of Stress-strain", fontsize=16)
    plt.tight_layout()
    plt.grid(False)
    # plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
    plt.savefig("{}.pdf".format(name))
    # plt.show()
    plt.close()

    return marker_shapes


if __name__ == "__main__":

    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    real_names = [
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

    newton_optim = [0.50 for i in range(len(names))]
    Adams_optim = [
        0.5013,
        0.5033,
        0.5005,
        0.5047,
        0.5058,
        0.4578,
        0.4510,
        0.3960,
        0.3823,
        0.5575,
        0.5578,
    ]
    values = newton_optim[5:] + Adams_optim[5:]
    params = real_names[5:] * 2
    optimizers = ["L-BFGS" for i in range(len(names[5:]))] + [
        "Adam" for i in range(len(names[5:]))
    ]

    df = pd.DataFrame(
        {
            "Values": values,
            "Optimizers": optimizers,
            "Params": params,
        }
    )
    print(df)
    
    sns.set(font_scale=2.0, style="white")
    sns.barplot(x="Params", y="Values", hue="Optimizers", data=df)
    sns.despine(top=False, right=False, left=False, bottom=False)
    ax = plt.gca()
    plt.ylim(0, 0.75)
    plt.locator_params(axis='y', nbins=3)
    plt.xlabel("", fontsize=18)
    plt.ylabel("", fontsize=18)
    plt.legend(ncol=2, frameon=False)
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("single-cyclic-kin.pdf")
    plt.show()
    plt.close()