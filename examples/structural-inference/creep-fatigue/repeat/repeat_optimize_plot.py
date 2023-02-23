#!/usr/bin/env python3

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path
import xarray as xr
import torch
from maker import make_model, load_subset_data
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as ra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from pyoptmat import optimize, experiments
# Don't care if integration fails
import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


def read_file(path, nsample, status="pvalues"):
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        cst = os.path.basename(f).split("-")[0]
        batch = os.path.basename(f).split("-")[2].split(".csv")[0]
        if float(batch) == nsample and cst == status:
            df = pd.read_csv(f)
            return df


def get_value(path, v, nsample=10, mu=False):
    df = read_file(path, nsample)
    df2 = df[v].loc[(df[v] > 0.0) & (df[v] < 1.0)]
    if mu:
        return df2.mean().tolist()
    else:
        return df2.values


def res(names, lists):
    for n, l in zip(names, lists):
        print("%s :" % (n), l)
    return names


def values(v, paths, batch=30, mu=False):
    data = []
    for path in paths:
        data.append(get_value(path, v, nsample=batch, mu=mu))
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
        if len(i) >= 1:
            for j in i:
                data.append(j)
    return data


def mean_values(names, real_names, path=None, batch=None):
    ps = []
    for p, rn in zip(names, real_names):
        result = values(p, path, batch=batch, mu=True)
        ps.append(result)
    return tolist(ps)


def Accuracy(pred, res, sf=0.05):
    correct = 0
    for i in range(res.shape[-1]):
        cond = torch.logical_and(
            pred[:, i] <= res[:, i] * (1 + sf),
            pred[:, i] >= res[:, i] * (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def New_Accuracy(pred, res, sf=0.15, tiny=torch.tensor(1.0e-10)):
    pred = (pred + tiny) / (res + tiny)
    correct = 0
    for i in range(res.shape[-1]):
        cond = torch.logical_and(
            pred[:, i] <= (1 + sf),
            pred[:, i] >= (1 - sf),
        )
        correct += ((cond).sum().item()) / res.shape[0]
    return correct * 100.0 / (res.shape[-1])


def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


def obtain_score(ics, scale=0.15, nsamples=20, full=False):
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples
    )    
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    model = optimize.DeterministicModel(make, names, ics)
    with torch.no_grad():
        pred = model(data, cycles, types, control)
    if full:
        return New_Accuracy(pred, results), pred, results, data
    else:
        return New_Accuracy(pred, results)


def cook_ics(pvs):
    data = []
    for i in range(5):
        data.append(torch.tensor(pvs[i]))
    data += [torch.tensor([pvs[5], pvs[6], pvs[7]])]
    data += [torch.tensor([pvs[8], pvs[9], pvs[10]])]
    return data


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/repeat/"
    bspath = path1 + path2

    path_b1 = bspath + "repeat/"
    path_b2 = bspath + "repeat2/"
    path_b3 = bspath + "repeat3/"
    path_b4 = bspath + "repeat4/"
    path_b5 = bspath + "repeat5/"

    paths = [path_b1, path_b2, path_b3, path_b4, path_b5]

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

    ics = [ra.uniform(0, 1) for i in range(30)]

    # batch size of 10
    mean_10 = mean_values(names, real_names, path=[path_b1], batch=10)

    # batch size of 20
    mean_20 = mean_values(names, real_names, path=[path_b2], batch=20)

    # batch size of 30
    mean_30_1 = mean_values(names, real_names, path=[path_b1], batch=30)
    mean_30_2 = mean_values(names, real_names, path=[path_b2], batch=30)
    mean_30_3 = mean_values(names, real_names, path=[path_b3], batch=30)

    mean_30_all = np.vstack(
        (np.array(mean_30_1), np.array(mean_30_2), np.array(mean_30_3))
    )

    mean_30 = list(mean_30_all.mean(axis=0))

    dist = False
    score = True
    opt_visualize = False
    
    if opt_visualize:
        ics_30 = cook_ics(mean_30)
        score_30, pred, results, data = obtain_score(ics_30, nsamples=30, full=True)
        
        plt.plot(data[2, :, 0].numpy(), results[:, 0].numpy(), lw=3, label="Actual")
        plt.plot(data[2, :, 0].numpy(), pred[:, 0].numpy(), lw=3, label="Prediction")

        ax = plt.gca()
        plt.xlabel("Strain", fontsize=23)
        plt.ylabel("Stress (MPa)", fontsize=23)
        plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        # plt.savefig("cyclic-loss.pdf")
        plt.show()
        plt.close()
    
    
    if score:
        
        # calculate score
        ics_10 = cook_ics(mean_10)
        score_10 = obtain_score(ics_10, nsamples=10)
        
        ics_20 = cook_ics(mean_20)
        score_20 = obtain_score(ics_20, nsamples=20)
        
        # ics_30 = cook_ics(mean_30)
        # score_30 = obtain_score(ics_30, nsamples=30)
        
        print(score_10, score_20)
        
        scores = np.array([score_10, score_20])
        df = pd.DataFrame(
            {
                "scores": scores,
                "values": np.array([10, 20]),
            }
        )
        print(df)
        sns.set(font_scale=1.5, style="white")

        p = sns.barplot(data=df, x="values", y="scores")
        ax = plt.gca()
        plt.ylabel("Strict Accuracy Score(%)", fontsize=23)
        plt.xlabel(r"$n_{sample}$", fontsize=23)
        plt.ylim(0, 100)
        plt.legend(loc="best", ncol=2, prop={"size": 20}, frameon=False)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        show_values(p, space=0)
        plt.tight_layout()
        plt.savefig("cyclic-accuracy-II.pdf")
        plt.show()
        plt.close()


    if dist:
        values = mean_10[5:] + mean_20[5:]# + mean_30[:5]
        bats = (
            [10 for i in range(len(names[5:]))]
            + [20 for i in range(len(names[5:]))]
            # + [30 for i in range(len(names[5:]))]
        )
        params = real_names[5:] * 2

        df = pd.DataFrame(
            {
                "params": params,
                "values": np.abs(np.array(values) - 0.5) * 100.0,
                "batchs": bats,
            }
        )
        print(df)
        sns.set(font_scale=1.5, style="white")

        sns.barplot(data=df, x="batchs", y="values", hue="params")
        ax = plt.gca()
        plt.xlabel(r"$n_{sample}$", fontsize=23)
        plt.ylabel("Optimization gap (%)", fontsize=23)
        plt.legend(loc="best", ncol=2, prop={"size": 20}, frameon=False)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("batch-bar-kin.pdf")
        plt.show()
        plt.close()
