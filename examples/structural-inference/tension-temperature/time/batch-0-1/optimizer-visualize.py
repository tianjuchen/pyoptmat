#!/usr/bin/env python3
import sys
import os.path
import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import matplotlib.pyplot as plt


def optimizer_compare(names, params, marker_shape, optimizer, fs=23):
    
    xs = np.arange(1, 6)
    for x, name, param in zip(xs, names, params):
        plt.plot(x, param, marker=marker_shape, markersize=fs, label="{}".format(name))
        
    plt.axhline(y = 0.5, color = 'k', lw=3, linestyle = '--')
    ax = plt.gca()
    plt.locator_params(axis='both', nbins=5)
    ax.set_xticklabels([0] + names, rotation=0, fontsize=18)
    plt.ylabel("Value", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    # plt.title("Scattering Predictive of Stress-strain", fontsize=16)
    plt.tight_layout()
    plt.grid(False)
    # plt.legend(prop={"size":14}, frameon=False, ncol=1, loc='best')
    plt.savefig("optim-compare-{}.pdf".format(optimizer))
    plt.show()
    plt.close()
    
    return marker_shapes

if __name__ == "__main__":

    adam_res = np.array([0.50, 0.51, 0.49, 0.50, 0.49])
    newton_res = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
    params = np.vstack((adam_res, newton_res))

    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    
    marker_shapes = ["o", "^"]
    optimizers = ["Adam", "Newton"]
    
    for param, maker_shape, optimizer in zip(params, marker_shapes, optimizers):
        _ = optimizer_compare(names, param, maker_shape, optimizer)