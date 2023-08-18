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

import xarray as xr
import torch, logging

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import tqdm

import pyro
from pyro.infer import SVI
import pyro.optim as optim

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC

import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd

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
def make(n, eta, s0, R, d, **kwargs):
    """
        Maker with Young's modulus fixed
    """
    return make_model(torch.tensor(0.5, device=device), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


def get_summary_table(
    posterior,
    sites,
    transforms={},
    diagnostics=False,
    group_by_chain=False,
):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    site_stats = {}

    for site_name in sites:
        marginal_site = posterior[site_name].cpu()

        if site_name in transforms:
            marginal_site = transforms[site_name](marginal_site)

        site_summary = mcmc.summary(
            {site_name: marginal_site}, prob=0.5, group_by_chain=group_by_chain
        )[site_name]
        if site_summary["mean"].shape:
            site_df = pd.DataFrame(site_summary)
        else:
            site_summary = {k: float(v) for k, v in site_summary.items()}
            site_df = pd.DataFrame(site_summary, index=[0])
        if not diagnostics:
            site_df = site_df.drop(["n_eff", "r_hat"], axis=1)
        site_stats[site_name] = site_df.astype(float).round(2)

    return site_stats

if __name__ == "__main__":

    start = time.time()
    # Number of vectorized time steps
    time_chunk_size = 40

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the priors
    names = ["n", "eta", "s0", "R", "d"]
    loc_loc_priors = [
        #torch.tensor(ra.random(), device=device) for i in range(len(names))
        torch.tensor(0.25, device=device) for i in range(len(names))
    ]
    loc_scale_priors = [torch.tensor(0.15, device=device) for i in range(len(names))]
    scale_scale_priors = [torch.tensor(0.15, device=device) for i in range(len(names))]

    eps = torch.tensor(1.0e-4, device=device)

    print("Initial parameter values:")
    print("\tloc loc\t\tloc scale\tscale scale")
    for n, llp, lsp, sp in zip(
        names, loc_loc_priors, loc_scale_priors, scale_scale_priors
    ):
        print("%s:\t%3.2f\t\t%3.2f\t\t%3.2f" % (n, llp, lsp, sp))
    print("")

    # 3) Create the actual model
    model = optimize.HierarchicalStatisticalModel(
            lambda *args, **kwargs: make(*args, block_size = time_chunk_size, **kwargs),
            names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps
    ).to(device)


    nuts_kernel = NUTS(model, adapt_step_size=True)

    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)

    mcmc.run(data, cycles, types, control, results)

    mcmc_results = mcmc.get_samples()

    print(mcmc_results)

    end = time.time()

    print("Elapsed time:", end - start)


    nuts_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    for site, values in mcmc.summary(nuts_samples).items():
        print("Site: {}".format(site))
        print(values, "\n")


    for n in names:
        logging.info(
                get_summary_table(
                    mcmc.get_samples(group_by_chain=True),
                    sites=[n],
                    player_names=player_names,
                    diagnostics=True,
                    group_by_chain=True,
                )[n]
            )


    chain = 4

    post_pyro_n = []
    post_pyro_eta = []
    post_pyro_s0 = []
    post_pyro_R = []
    post_pyro_d = []
    for i in range(chain):
        pyro.get_param_store().clear()
        for j, (t, _) in tqdm(enumerate(mcmc_run._traces(data)), total=201):
            post_trace_n = t.nodes['n']['value']
            post_pyro_n.append(post_trace_n.data.numpy())
            
            post_trace_eta = t.nodes['eta']['value']
            post_pyro_eta.append(post_trace_n.data.numpy())
            
            post_trace_s0 = t.nodes['s0']['value']
            post_pyro_s0.append(post_trace_n.data.numpy())
            
            post_trace_R = t.nodes['R']['value']
            post_pyro_R.append(post_trace_n.data.numpy())
            
            post_trace_d = t.nodes['d']['value']
            post_pyro_d.append(post_trace_n.data.numpy())
    
    post_pyro_n = np.asarray(post_pyro_n)
    post_pyro_eta = np.asarray(post_pyro_eta)
    post_pyro_s0 = np.asarray(post_pyro_s0)
    post_pyro_R = np.asarray(post_pyro_R)
    post_pyro_d = np.asarray(post_pyro_d)
    
    
    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(post_pyro_n.squeeze());
    plt.tight_layout();
    plt.show()
    plt.close()
    
    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(post_pyro_eta.squeeze());
    plt.tight_layout();
    plt.show()
    plt.close()
    
    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(post_pyro_s0.squeeze());
    plt.tight_layout();
    plt.show()
    plt.close()
    
    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(post_pyro_R.squeeze());
    plt.tight_layout();
    plt.show()
    plt.close()
    
    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(post_pyro_d.squeeze());
    plt.tight_layout();
    plt.show()
    plt.close()