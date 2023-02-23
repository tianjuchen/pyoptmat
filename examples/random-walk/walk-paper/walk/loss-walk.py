#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
import pandas as pd
import xarray as xr
import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_file(path, fn=None):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        file_name = os.path.basename(f).split(".txt")[0]
        if file_name == fn:
            print(file_name)
            df = pd.read_csv(f, header=None)
            return df


def make_loss(data, total_length=1000):
    len2 = total_length - len(list(data))
    data2 = [data[-1] for i in range(len2)]
    return list(data) + data2


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "random-walk/walk-paper/walk/"

    p = "loss-history-rw"

    loss_walk = load_file(path1 + path2, fn=p)

    ax = plt.gca()
    plt.semilogy(loss_walk, ls="-", lw=4, label="Training")
    plt.xlabel("Step", fontsize=23)
    plt.ylabel("Loss", fontsize=23)
    # plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
    plt.tick_params(axis="both", which="major", labelsize=23)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("walk-loss.pdf")
    plt.show()
    plt.close()
