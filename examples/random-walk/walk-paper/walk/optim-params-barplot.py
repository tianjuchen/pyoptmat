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


if __name__ == "__main__":

    n, eta, s0, R, d = 0.4368, 0.4507, 0.7640, 0.5068, 0.5049
    C1, C2, C3 = 0.5003, 0.5211, 0.4706
    g1, g2, g3 = 0.3565, 0.7994, 0.4607

    params = [n, eta, s0, R, d, C1, C2, C3, g1, g2, g3]
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

    df = {
        "params": params,
        "names": real_names,
    }

    print(df)

    sns.set(font_scale=1.5, style="white")
    sns.barplot(data=df, x="names", y="params")
    plt.legend(loc="upper right", frameon=False, ncol=1, prop={"size": 20})
    ax = plt.gca()
    plt.xlabel("", fontsize=23)
    plt.ylabel("Value", fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("rw-optim-compare.pdf")
    plt.show()
    plt.close()
