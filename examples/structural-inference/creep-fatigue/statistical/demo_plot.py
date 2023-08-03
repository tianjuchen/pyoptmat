#!/usr/bin/env python3

import sys, glob
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import xarray as xr
import torch
import pandas as pd
import numpy as np

from maker import make_model, load_subset_data

from pyoptmat import optimize, experiments
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import RcParams

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on this on the cpu
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)



def read_file(path, file_name, ftype=".csv"):
    fnames = glob.glob(path + "*" + ftype)
    for f in fnames:
        ffname = os.path.basename(f).split(ftype)[0]
        if ffname == file_name:
            df = pd.read_csv(f)
            return df


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)



def convert_results(data, res, N=5):
    mes = []
    for i in range(data.shape[-1]):
        mes.append(data[2, :, i].max())

    need_global_ind = []
    for i in range(res.shape[-1]):
        ind = data[2, :, i] == mes[i].item()
        # print(ind[ind==True].shape)
        need_local_ind = []
        for j in range(len(ind) - 1):
            if ind[j-1] == False and ind[j] == True:
                need_local_ind.append(j)
        need_global_ind.append(need_local_ind)

    new_res = torch.zeros(N, res.shape[-1])
    for i in range(res.shape[-1]):
        ind = need_global_ind[i]
        new_res[:, i] = res[ind, i]
    
    return new_res




def create_input(mus, stds):

    ics = []
    iscales = []
    for i, (m, s) in enumerate(zip(mus[:5], stds[:5])):
        ics.append(torch.tensor(m, device=device))
        iscales.append(torch.tensor(s, device=device))
        
    ics += torch.tensor([mus[5:8]], device=device)
    ics += torch.tensor([mus[8:11]], device=device)
    
    iscales += torch.tensor([stds[5:8]], device=device)
    iscales += torch.tensor([stds[8:11]], device=device)

    return ics , iscales
    
if __name__ == "__main__":


    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/"
    path2 = "structural-inference/creep-fatigue/statistical/repeat/"

    path = path1 + path2

    fns = "res-30-"
    scales = ["0.05", "0.1", "0.15"]
    vns = ["mu", "std"]

    names = ["n", "eta", "s0", "R", "d", "C1", "C2", "C3", "g1", "g2", "g3"]
    
    scale = "0.15"
    
    df = read_file(path, fns + scale)
    mus = []
    stds = []
    for n in names:
        vn = n + "_" + vns[0]
        sn = n + "_" + vns[1]
        print("%s is %.2f, %.2f" %(n, df[vn].mean(), df[sn].mean()))
        mus.append(df[vn].mean())
        stds.append(df[sn].mean())


    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    N = 5
    scale = float(scale)
    nsamples = 20  # 10 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    results = convert_results(data, results)
    count = torch.arange(5) + 1

    ics, iscale = create_input(mus, stds)

    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    sampler = optimize.StatisticalModel(
        make,
        names,
        ics,
        iscale,
        torch.tensor(1.0e-4),
    )

    plt.style.use(latex_style_times)
    plt.figure()
    plt.plot(count, results[:, :nsamples].cpu(), "k--")
    ax = plt.gca()
    
    nsamples = 25
    alpha = 0.05 / 2

    cal_results = torch.zeros(nsamples, data.shape[1])
    stress_results = torch.zeros(nsamples, N)


    for i in trange(nsamples):
        cal_results[i, :] = sampler(data, cycles, types, control)[:, 0]  
        stress_results[i, :] = convert_results(data[:, :, :1], cal_results[i, :].unsqueeze(-1))[:, 0]
    
    mean_result = torch.mean(stress_results, 0)
    sresults, _ = torch.sort(stress_results, 0)
    min_result = sresults[int(alpha * nsamples), :]
    max_result = sresults[int((1 - alpha) * nsamples), :]

    
    (l,) = plt.plot(count, mean_result, lw=4, color="k")
    p = plt.fill_between(count, min_result, max_result, alpha=0.5, color="k")

    plt.legend(
        [
            Line2D([0], [0], color="k", ls="--"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="k", edgecolor=None, alpha=0.5),
        ],
        ["Experimental data", "Model average", "Model 95\% prediction interval"],
        loc="best",
        prop={'size': 18}
    )

    plt.xlabel("Cycles", fontsize=30)
    plt.ylabel("Stress (MPa)", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.locator_params(axis="both", nbins=4)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("cyclic-confidence-{}.pdf".format(float(scale)))
    # plt.show()
    plt.close()
