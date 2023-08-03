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
import torch

from maker import downsample
from pyoptmat import models, flowrules, hardening, optimize, scaling
from pyoptmat.temperature import ConstantParameter as CP
from pyoptmat import optimize, experiments
from tqdm import tqdm

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


def make_model(E, n, eta, s0, R, d, scale_functions=None, **kwargs):

    if scale_functions is None:
        E_scale = lambda x: x
        n_scale = lambda x: x
        eta_scale = lambda x: x
        s0_scale = lambda x: x
        R_scale = lambda x: x
        d_scale = lambda x: x
        
    else:
        E_scale = scale_functions[0]
        n_scale = scale_functions[1]
        eta_scale = scale_functions[2]
        s0_scale = scale_functions[3]
        R_scale = scale_functions[4]
        d_scale = scale_functions[5]

    iso = hardening.VoceIsotropicHardeningModel(
        CP(R, p_scale=R_scale),
        CP(d, p_scale=d_scale),
    )
    kin = hardening.NoKinematicHardeningModel()
    fr = flowrules.IsoKinViscoplasticity(
        CP(n, p_scale=n_scale),
        CP(eta, p_scale=eta_scale),
        CP(s0, p_scale=s0_scale),
        iso,
        kin,
    )
    model = models.InelasticModel(
        CP(E, p_scale=E_scale), fr
    )
    return models.ModelIntegrator(model, **kwargs)

true = [
    torch.tensor(150000.0, device=device),
    torch.tensor(7.0, device=device), 
    torch.tensor(300.0, device=device), 
    torch.tensor(50.0, device=device), 
    torch.tensor(200.0, device=device), 
    torch.tensor(5.0, device=device),     
]
lbs = [t * 0.5 for t in true]
ubs = [t * 1.5 for t in true]
scales_fn = [scaling.BoundedScalingFunction(l, u) for l, u in zip(lbs, ubs)]

def make(p, **kwargs):
    """
    Maker with Young's modulus fixed
    """
    return make_model(p[0], p[1], p[2], p[3], p[4], p[5], scale_functions=scales_fn, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.05
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    
    # 2) Setup names for each parameter and the priors
    names = ["p"]
    loc_loc_priors = [torch.tensor([0.2] * len(true), device=device)]
    loc_scale_priors = [torch.ones(len(true), device=device) * 0.15]
    scale_scale_priors = [torch.ones(len(true), device=device) * 0.15]

    eps = torch.tensor(1.0e-4, device=device)

    print("Initial parameter values:")
    print("\tloc loc\t\tloc scale\tscale scale")
    for n, llp, lsp, sp in zip(
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
    ):
        print(n)
        print(llp)
        print(lsp)
        print(sp)
        # print("%s:\t%3.2f\t\t%3.2f\t\t%3.2f" % (n, llp, lsp, sp))
    print("")

    # 3) Create the actual model
    model = optimize.HierarchicalMultivariateModel(
        make, names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps
    ).to(device)

    # 4) Get the guide
    guide = model.make_guide()
    pyro.clear_param_store()

    # 5) Setup the optimizer and loss
    lr = 1.0e-2
    g = 1.0
    niter = 200
    lrd = g ** (1.0 / niter)
    num_samples = 1

    optimizer = optim.ClippedAdam({"lr": lr, "lrd": lrd})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = SVI(model, guide, optimizer, loss=ls)

    # print("Pyro Inferred Parameters are:")
    # for name, value in pyro.get_param_store().items():
        # print(name, pyro.param(name))
    # print("")
    # sys.exit("stop")

    # 6) Infer!
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data, cycles, types, control, results)
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)

    # 7) Print out results
    print("")
    print("Inferred distributions:")
    print("\tloc\t\tscale")
    for n in names:
        s = pyro.param(n + model.scale_suffix + model.param_suffix).data
        m = pyro.param(n + model.loc_suffix + model.param_suffix).data
        print("%s:\t%3.2f/0.50\t%3.2f/%3.2f" % (n, m, s, scale))
    print("")

    # 8) Plot convergence
    plt.figure()
    plt.loglog(loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
