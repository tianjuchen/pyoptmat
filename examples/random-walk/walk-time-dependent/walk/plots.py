#!/usr/bin/env python3
import sys
import os.path
import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import matplotlib.pyplot as plt


def optimization_random_walk(names, params, fname):

    for param, name in zip(params, names):
        plt.plot(param, 0.5, marker="o", markersize=18, label=name)

    plt.axvline(x=0.5, color="k", linestyle="--", lw=3)
    
    plt.xlim([0, 1])
    plt.legend(loc="upper left", frameon=False, ncol=2, prop={"size": 20})
    ax = plt.gca()
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(fname))
    plt.show()
    plt.close()

    return params


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

    _ = optimization_random_walk(real_names[:5], params[:5], "isotropic")
