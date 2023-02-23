#!/usr/bin/env python3
import sys
import os.path
import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import matplotlib.pyplot as plt


def learning_rate_effect(name, params, lrs, fs=18):

    marker_shapes = ["o", "^", "*"]

    for param, lr, marker_shape in zip(params, lrs, marker_shapes):
        plt.plot(param, lr, marker=marker_shape, markersize=fs)
        plt.axvline(x = 0.5, color = 'k', lw=3, linestyle = '--')

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

    lrs = np.array([1.0e-1, 1.0e-2, 1.0e-3])
    ns = np.array([0.50, 0.46, 0.81])
    etas = np.array([0.50, 0.51, 0.18])
    s0s = np.array([0.50, 0.56, 0.42])
    Rs = np.array([0.50, 0.50, 0.58])
    ds = np.array([0.50, 0.49, 0.47])
    params = [ns, etas, s0s, Rs, ds]
    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    
    for name, param in zip(names, params):
        _ = learning_rate_effect(name, param, lrs)