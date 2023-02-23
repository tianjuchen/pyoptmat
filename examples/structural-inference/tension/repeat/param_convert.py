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

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/tension/repeat/"
    bspath = path1 + path2

    path_b1 = bspath + "repeat/"
    path_b2 = bspath + "repeat2/"
    path_b3 = bspath + "repeat3/"
    path_b4 = bspath + "repeat4/"
    path_b5 = bspath + "repeat5/"

    paths = [path_b1, path_b2, path_b3, path_b4, path_b5]

    names = ["n", "eta", "s0", "R", "d"]
    real_names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    
    ics = [ra.uniform(0, 1) for i in range(30)]
    
    for i, (p, rn) in enumerate(zip(names, real_names)):
        result = values(p, paths, batch=30)
        ps = tolist(result)
        ics = [ra.uniform(0, 1) for i in range(30)]
        
        ltype1 = ["initial" for i in range(len(ics))]
        ltype2 = ["optimize" for i in range(len(ps))]
        
        df = pd.DataFrame(
            {
                "params": ics+ps,
                "class": ltype1+ltype2,
            }
        )
        sns.set(font_scale=1.7, style="white")
        sns.histplot(data=df, x="params", hue="class", element="step")
        ax = plt.gca()
        plt.xlabel("{}".format(rn), fontsize=23)
        plt.ylabel("Frequency", fontsize=23)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        if i == len(names) - 2:
            pass
        else:
            plt.legend(loc="best", prop={'size': 23}, frameon=False)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("batch-hist-{}.pdf".format(p))
        # plt.show()
        plt.close()
        
