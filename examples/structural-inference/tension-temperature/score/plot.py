#!/usr/bin/env python3

import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_file(path, ltype="test", ref="score", ratio=0.2):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        learn = os.path.basename(f).split("-")[0]
        res = os.path.basename(f).split("-")[1].split("-")[0]
        percent = os.path.basename(f).split("-")[2].split(".txt")[0]
        if learn == ltype and res == ref:
            if float(percent) == ratio:
                df = pd.read_csv(f)
                return df


def final_value(df1, df2, df3, df4, N1=1, N2=1, N3=1, N4=1):
    
    y = [
        df4.to_numpy()[-1, 0]/float(N4),
        df3.to_numpy()[-1, 0]/float(N3),
        df2.to_numpy()[-1, 0]/float(N2),
        df1.to_numpy()[-1, 0]/float(N1),
    ]
    
    return y

if __name__ == "__main__":

    # Load moose calculated results
    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/"
    path2 = "examples/structural-inference/tension/score/"
    path = path1 + path2

    df1 = read_file(path, ltype="train", ref="score", ratio=0.2)
    df2 = read_file(path, ltype="train", ref="score", ratio=0.3)
    df3 = read_file(path, ltype="train", ref="score", ratio=0.4)
    df4 = read_file(path, ltype="train", ref="score", ratio=0.5)

    df5 = read_file(path, ltype="test", ref="score", ratio=0.2)
    df6 = read_file(path, ltype="test", ref="score", ratio=0.3)
    df7 = read_file(path, ltype="test", ref="score", ratio=0.4)
    df8 = read_file(path, ltype="test", ref="score", ratio=0.5)



    y_train = final_value(df1, df2, df3, df4, N1=1, N2=1, N3=1, N4=1)
    y_test = final_value(df5, df6, df7, df8, N1=1, N2=1, N3=1, N4=1)
    x = np.array([0.5, 0.6, 0.7, 0.8]) * 30.0

    # Plot results against each case
    ax = plt.gca()
    plt.plot(x, y_train, lw=3, label="training")
    plt.plot(x, y_test, lw=3, label="validation")
    plt.xlabel(r"$n_{sample}$", fontsize=16)
    plt.ylabel(r"$Accuracy$", fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="best", frameon=False, prop={"size": 16})
    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    plt.savefig(path + "accuracy-batch.pdf")
    plt.show()
    plt.close()
