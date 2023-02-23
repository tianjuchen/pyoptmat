#!/usr/bin/env python3
import sys, glob
import os.path
import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def read_file(path, optimizer=None, dtype=None):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        out = os.path.basename(f).split("-")[0]
        alg = os.path.basename(f).split("-")[1].split(".txt")[0]
        if out == dtype and alg == optimizer:
            df = pd.read_csv(f)
            return df


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/single/"

    path = path1 + path2

    loss_adam = read_file(path, optimizer="adam", dtype="loss")
    time_adam = read_file(path, optimizer="adam", dtype="time")

    loss_newton = read_file(path, optimizer="newton", dtype="loss")
    time_newton = read_file(path, optimizer="newton", dtype="time")


    print(time_adam.values[30], time_newton.values[1])
    
    ax = plt.gca()
    plt.plot(time_adam, loss_adam, lw=3, label="Adam")
    plt.plot(time_newton, loss_newton, lw=3, label="L-BFGS")


    plt.locator_params(axis="both", nbins=4)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend(loc="best", ncol=1, frameon=False, prop={"size": 23})
    ax.tick_params(axis="both", which="major", labelsize=23)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("single-cyclic-loss.pdf")
    plt.show()
    plt.close()
    