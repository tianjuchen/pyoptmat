#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path

import numpy as np
import numpy.random as ra
from torch.nn import Parameter
import xarray as xr
import torch

from maker import make_model, load_subset_data
from pyoptmat import models, flowrules, hardening, experiments, optimize, utility
from pyoptmat.temperature import ConstantParameter as CP

from pyoptmat import optimize, experiments
from tqdm import tqdm

import matplotlib.pyplot as plt

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

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
make = lambda p: models.ModelIntegrator(
    models.InelasticModel(
        CP(torch.tensor(150000.0)),
        flowrules.IsoKinViscoplasticity(
            CP(p[0]),
            CP(p[1]),
            CP(p[2]),
            hardening.VoceIsotropicHardeningModel(CP(p[3]), CP(p[4])),
            hardening.ChabocheHardeningModel(CP(p[5]), CP(p[6])),
        ),
    )
)

extract_grad = lambda m: [
    m.model.flowrule.n.pvalue.grad.numpy(),
    m.model.flowrule.eta.pvalue.grad.numpy(),
    m.model.flowrule.s0.pvalue.grad.numpy(),
    m.model.flowrule.isotropic.R.pvalue.grad.numpy(),
    m.model.flowrule.isotropic.d.pvalue.grad.numpy(),
    m.model.flowrule.kinematic.C.pvalue.grad.numpy(),
    m.model.flowrule.kinematic.g.pvalue.grad.numpy(),
]


def differ(mfn, p0, eps=1.0e-6):
    v0 = mfn(p0).numpy()

    puse = p0.numpy()

    result = np.zeros(puse.shape)

    for ind, val in np.ndenumerate(puse):
        dp = np.abs(val) * eps
        if dp < eps:
            dp = eps
        pcurr = np.copy(puse)
        pcurr[ind] += dp
        v1 = mfn(torch.tensor(pcurr)).numpy()
        result[ind] = (v1 - v0) / dp

    return result


def simple_diff(fn, p0):
    res = []
    for i in range(len(p0)):

        def mfn(pi):
            ps = [pp for pp in p0]
            ps[i] = pi
            return fn(ps)

        res.append(differ(mfn, p0[i]))

    return res


if __name__ == "__main__":

    scale = 0.15
    nsamples = 1  # 20 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    def adjoint_grad(fn, p):
        bmodel = fn([Parameter(pi) for pi in p])
        res = torch.norm(bmodel.solve_both(data[0], data[1], data[2], control))
        res.backward()
        grad = extract_grad(bmodel)
        return grad

    def fd_grad(fn, p):
        with torch.no_grad():
            ngrad = simple_diff(
                lambda p: torch.norm(
                    fn(p).solve_both(data[0], data[1], data[2], control)
                ),
                p,
            )
        return ngrad

    ics = [
        torch.tensor(7.0),
        torch.tensor(300.0),
        torch.tensor(50.0),
        torch.tensor(200.0),
        torch.tensor(5.0),
        torch.tensor([10000.0, 5000.0, 500.0]),
        torch.tensor([200.0, 150.0, 5.0]),
    ]

    grad = adjoint_grad(make, ics)
    ngrad = fd_grad(make, ics)
    for i, (p1, p2) in enumerate(zip(grad, ngrad)):
        print(i, p1, p2)
        np.allclose(p1, p2, rtol=1e-4)
