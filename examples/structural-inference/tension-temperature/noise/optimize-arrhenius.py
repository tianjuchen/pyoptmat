#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path

import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, downsample, grid_model

from pyoptmat import optimize, experiments, scaling
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

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0


def pgrid(p, T, Q=-100.0):
    return p * torch.exp(-Q/T)

Tcontrol = torch.tensor([25.0, 200.0, 300.0, 400.0, 500.0, 600.0]) + 273.15
trues = [E_true, n_true, eta_true, s0_true, R_true, d_true]
scale_fn = [
    scaling.BoundedScalingFunction(torch.tensor(p) * 0.5, torch.tensor(p) * 1.5)
    for p in trues
]    

def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, 
        scale_functions=scale_fn,
        use_adjoint=True, miter=10,
        device=device, **kwargs).to(
        device
    )

def train(repeat_time, nsamples=1, scale=0.00):

    ivalues = []
    pvalues = []

    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    print("temperature levels are:", data[1, 0, :])
    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d"]

    ics = [torch.tensor(ra.uniform(0.01, 0.99)) for i in range(len(names))]
    
    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s:" % (n),  ic)
        ivalues.append(ic.data)
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    niter = 300
    lr = 1.0e-2
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        with torch.no_grad():
          for param in model.parameters():
            param.clamp_(0.01, 0.99)
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
        for param in model.parameters():
            print(param.data, param.grad)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))
        pvalues.append(getattr(model, n).data)

    # 8) Save the convergence history
    np.savetxt("loss-history-arrhenius-adam-{}.txt".format(repeat_time), loss_history)

    return ivalues, pvalues

if __name__ == "__main__":

    nrepeat, batch = 20, 30
    ns = []
    etas = []
    s0s = []
    Rs = []
    ds = []

    for i in range(nrepeat):
        ivalues, pvalues = train(i, nsamples=batch, scale=0.00)
        ns.append(pvalues[0])
        etas.append(pvalues[1])
        s0s.append(pvalues[2])
        Rs.append(pvalues[3])
        ds.append(pvalues[4])

    data = pd.DataFrame(
        {
            "n": np.array(ns),
            "eta": np.array(etas),
            "s0": np.array(s0s),
            "R": np.array(Rs),
            "d": np.array(ds),
        }
    )
    data.to_csv("pvalues-{}-{}.csv".format(nrepeat, batch))
