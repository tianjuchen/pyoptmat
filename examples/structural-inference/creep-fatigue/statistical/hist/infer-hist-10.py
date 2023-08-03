#!/usr/bin/env python3

"""
    Tutorial example of training a statistical model to tension test data
    from from a known distribution.
"""

import sys
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import numpy.random as ra
import numpy as np
import xarray as xr
import torch

from maker import make_model, load_subset_data

from pyoptmat import optimize, experiments
from tqdm import tqdm
import pandas as pd
import pyro
from pyro.infer import SVI
import pyro.optim as optim

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
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


def train(time_chunk_size=10, scale=0.15, nsamples=30):
    n_mu = []
    eta_mu = []
    s0_mu = []
    R_mu = []
    d_mu = []
    C1_mu = []
    C2_mu = []
    C3_mu = []
    g1_mu = []
    g2_mu = []
    g3_mu = []

    n_std = []
    eta_std = []
    s0_std = []
    R_std = []
    d_std = []
    C1_std = []
    C2_std = []
    C3_std = []
    g1_std = []
    g2_std = []
    g3_std = []

    # 1) Load the data for the variance of interest,
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    # 2) Setup names for each parameter and the priors
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    loc_loc_priors = [
        torch.tensor(ra.uniform(0, 1), device=device) for i in range(5)
    ] + [
        torch.tensor(ra.uniform(0, 1, size=(3,)), device=device),
        torch.tensor(ra.uniform(0, 1, size=(3,)), device=device),
    ]
    loc_scale_priors = [0.15 * torch.ones_like(p) for p in loc_loc_priors]
    scale_scale_priors = [0.5 * torch.ones_like(p) for p in loc_loc_priors]

    eps = torch.tensor(1.0e-4, device=device)

    print("Initial parameter values:")
    print("\tloc loc\t\t\t\tloc scale\t\t\tscale scale")
    for n, llp, lsp, sp in zip(
        names, loc_loc_priors, loc_scale_priors, scale_scale_priors
    ):
        print(n + "\t" + str(llp.data) + "\t" + str(lsp.data) + "\t" + str(sp.data))
    print("")

    # 3) Create the actual model
    model = optimize.HierarchicalStatisticalModel(
        lambda *args, **kwargs: make(*args, block_size=time_chunk_size, **kwargs),
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        eps,
        include_noise=False,
    ).to(device)

    # 4) Get the guide
    guide = model.make_guide()

    # 5) Setup the optimizer and loss
    lr = 1.0e-2
    g = 1.0
    niter = 500
    lrd = g ** (1.0 / niter)
    num_samples = 1

    optimizer = optim.Adam({"lr": lr})
    # optimizer = optim.ClippedAdam({"lr": lr, "lrd": lrd})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = SVI(model, guide, optimizer, loss=ls)

    # 6) Infer!
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data, cycles, types, control, results)
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)
        for i, n in enumerate(names):
            if i <= 4:
                s = pyro.param(n + model.scale_suffix + model.param_suffix).item()
                m = pyro.param(n + model.loc_suffix + model.param_suffix).item()
                if i == 0:
                    n_mu.append(m)
                    n_std.append(s)
                elif i == 1:
                    eta_mu.append(m)
                    eta_std.append(s)
                elif i == 2:
                    s0_mu.append(m)
                    s0_std.append(s)
                elif i == 3:
                    R_mu.append(m)
                    R_std.append(s)
                elif i == 4:
                    d_mu.append(m)
                    d_std.append(s)
            else:
                s = pyro.param(n + model.scale_suffix + model.param_suffix)
                m = pyro.param(n + model.loc_suffix + model.param_suffix)
                if i == 5:
                    C1_mu.append(m[0].item())
                    C1_std.append(s[0].item())
                    C2_mu.append(m[1].item())
                    C2_std.append(s[1].item())
                    C3_mu.append(m[2].item())
                    C3_std.append(s[2].item())
                elif i == 6:
                    g1_mu.append(m[0].item())
                    g1_std.append(s[0].item())
                    g2_mu.append(m[1].item())
                    g2_std.append(s[1].item())
                    g3_mu.append(m[2].item())
                    g3_std.append(s[2].item())

    # 7) Print out results
    print("")
    print("Inferred distributions:")
    print("\tloc\t\tscale")
    for n in names:
        s = pyro.param(n + model.scale_suffix + model.param_suffix).data
        m = pyro.param(n + model.loc_suffix + model.param_suffix).data
        print(n + "\t" + str(m) + "\t" + str(s))
    print("")

    # 8) Plot convergence
    np.savetxt("loss-history-hist-{}.txt".format(scale), loss_hist)

    return (
        n_mu,
        eta_mu,
        s0_mu,
        R_mu,
        d_mu,
        C1_mu,
        C2_mu,
        C3_mu,
        g1_mu,
        g2_mu,
        g3_mu,
        n_std,
        eta_std,
        s0_std,
        R_std,
        d_std,
        C1_std,
        C2_std,
        C3_std,
        g1_std,
        g2_std,
        g3_std,
    )


if __name__ == "__main__":
    (
        n_mu,
        eta_mu,
        s0_mu,
        R_mu,
        d_mu,
        C1_mu,
        C2_mu,
        C3_mu,
        g1_mu,
        g2_mu,
        g3_mu,
        n_std,
        eta_std,
        s0_std,
        R_std,
        d_std,
        C1_std,
        C2_std,
        C3_std,
        g1_std,
        g2_std,
        g3_std,
    ) = train(time_chunk_size=10, scale=0.1, nsamples=20)

    data = pd.DataFrame(
        {
            "n_mu": n_mu,
            "n_std": n_std,
            "eta_mu": eta_mu,
            "eta_std": eta_std,
            "s0_mu": s0_mu,
            "s0_std": s0_std,
            "R_mu": R_mu,
            "R_std": R_std,
            "d_mu": d_mu,
            "d_std": d_std,
            "C1_mu": C1_mu,
            "C1_std": C1_std,
            "C2_mu": C2_mu,
            "C2_std": C2_std,
            "C3_mu": C3_mu,
            "C3_std": C3_std,
            "g1_mu": g1_mu,
            "g1_std": g1_std,
            "g2_mu": g2_mu,
            "g2_std": g2_std,
            "g3_mu": g3_mu,
            "g3_std": g3_std,
        }
    )

    data.to_csv("hist-0.1-20.csv")

