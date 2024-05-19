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

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import tqdm

import pyro
from pyro.infer import SVI
import pyro.optim as optim
from pyro.contrib.autoguide import (
    AutoDelta,
    init_to_mean,
    init_to_feasible,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
)
import pyro.distributions as dist

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
def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5, device=device),
        n,
        eta,
        s0,
        R,
        d,
        device=device,
        use_adjoint=False,
        **kwargs
    ).to(device)


p_prior_loc = torch.tensor(ra.random(5), device=device)
p_prior_cov = torch.eye(p_prior_loc.shape[0], device=device) * 0.5
p_prior_loc_scale = torch.ones_like(p_prior_loc) * 0.01
p_prior_scale = torch.ones_like(p_prior_loc) * 0.01

eps = torch.tensor(1.0e-4, device=device)


def model(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):

    options = dict(dtype=exp_data.dtype, device=exp_data.device)

    p_mean = pyro.sample(
        "p_mean",
        dist.Normal(
            torch.tensor(ra.random(5), device=device),
            torch.tensor([0.01] * 5, device=device),
        ).to_event(p_prior_loc.dim()),
    )
    p_scale = pyro.sample(
        "p_scale", dist.HalfCauchy(torch.ones(5, **options) * 0.05).to_event(p_prior_loc.dim())
    )

    concentration = torch.ones((), **options)

    L_omega = pyro.sample("L_omega", dist.LKJCholesky(5, concentration))

    L_Omega = torch.mm(
        torch.diag(p_scale.sqrt()), L_omega
    )  # torch.bmm(p_scale.sqrt().diag_embed(), L_omega)
    p = pyro.sample(
        "p",
        dist.MultivariateNormal(loc=p_mean, scale_tril=L_Omega),
    )

    # noise = pyro.sample("eps", dist.HalfNormal(eps))
    noise = eps

    with pyro.plate("trials", exp_data.shape[2]):

        # p = pyro.sample("p", dist.Normal(p_mean, p_scale).to_event(p_prior_loc.dim()))
        bmodel = make(*p)
        predictions = bmodel.solve_both(
            exp_data[0], exp_data[1], exp_data[2], exp_control
        )
        # Process the results
        results = experiments.convert_results(
            predictions[:, :, 0], exp_cycles, exp_types
        )
        # Sample!
        with pyro.plate("time", exp_data.shape[1]):
            pyro.sample("obs", dist.Normal(results, noise), obs=exp_results)


guide = AutoDelta(model, init_loc_fn=init_to_feasible)


def eyelist(data):
    newdata = []
    for i in range(data.shape[-1]):
        newdata.append(list(data[:, i]))
    return newdata


def usereye(scale, offvalue, size=5):
    return np.eye(size) * scale + (np.ones((size, size)) - np.eye(size)) * offvalue


if __name__ == "__main__":

    scale = 0.15
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f-offvalue.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    
    # 5) Setup the optimizer and loss
    lr = 1.0e-2
    g = 1.0
    niter = 200
    lrd = g ** (1.0 / niter)
    num_samples = 1

    optimizer = optim.ClippedAdam({"lr": lr, "lrd": lrd})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = SVI(model, guide, optimizer, loss=ls)

    # 6) Infer!
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data, cycles, types, control, results)
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)
        for name, value in pyro.get_param_store().items():
            print(name, pyro.param(name))
        print("")
    
        pyro.get_param_store().save("lkj-ad.pt")
    
    # 7) Save the history for future plot revision
    np.savetxt("lkj-ad-loss-history.txt", loss_hist)

    # 8) Plot convergence
    plt.figure()
    plt.loglog(loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("loss-history-lkj-AD.pdf")
    # plt.show()
    plt.close()
    
    pyro.get_param_store().load("lkj-ad.pt")


    loc = np.array([0.5] * 5)
    # scale = np.zeros((5, 5))
    # scale = eyelist(np.eye(5) * scale)
    scale = eyelist(usereye(scale, 0.05))  # eyelist(np.eye(5) * scale)
    truth = np.random.multivariate_normal(loc, scale, 5000)

    loc_prediction = pyro.param("AutoDelta.p_mean").data.numpy()
    L_omega = pyro.param("AutoDelta.L_omega")
    L_Omega = torch.mm(
        torch.diag(pyro.param(
        "AutoDelta.p_scale"
    ).sqrt()), L_omega
    )
    
    scale_tril = L_Omega.data.numpy()
    scale_prediction = scale_tril @ scale_tril.T
    prediction = np.random.multivariate_normal(loc_prediction, scale_prediction, 5000)

    #plt.scatter(truth[:, 0], truth[:, 1], c="r", label="truth", alpha=0.1)
    plt.scatter(
        prediction[:, 0], prediction[:, 1], c="b", label="prediction", alpha=0.3
    )
    plt.scatter(truth[:, 0], truth[:, 1], c="r", label="truth", alpha=0.1)
    plt.legend(loc="lower left")
    plt.savefig("lkj-scatter-AD.pdf")
    plt.show()
    plt.close()
