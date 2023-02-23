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
import numpy as np
import xarray as xr
import torch

from maker import make_model, downsample, power_model

from pyoptmat import optimize, experiments, scaling
from tqdm import tqdm

import matplotlib.pyplot as plt
import time

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
def make(n, eta, s0, A, p, scale_fn, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return power_model(
        torch.tensor(0.5),
        n,
        eta,
        s0,
        A,
        p,
        scale_functions=scale_fn,
        device=device,
        **kwargs
    ).to(device)


# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0
A_true = 400.0
p_true = 0.5
sf = 0.9

trues = [E_true, n_true, eta_true, s0_true, A_true, p_true]


if __name__ == "__main__":

    start_time = time.time()
    time_list = [start_time]
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.15
    nsamples = 20  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )
    
    indices = torch.randint(0, results.shape[-1], (10,))
    print(indices)

    data = data[:, :, indices]
    results = results[:, indices]
    cycles = cycles[:, indices]
    types = types[indices]
    control = control[indices]
    
    # amount = 1
    # data = data[:, :, :amount]
    # results = results[:, :amount]
    # cycles = cycles[:, :amount]
    # types = types[:amount]
    # control = control[:amount]

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "A", "p"]
    # ics = torch.tensor([ra.uniform(0.5, 0.5) for i in range(len(names))], device=device)

    # ics = [
        # torch.tensor(0.40),
        # torch.tensor(0.50),
        # torch.tensor(0.14),
        # torch.tensor(0.39),
        # torch.tensor(0.36),
    # ]

    ics = [
        torch.tensor(0.44),
        torch.tensor(0.49),
        torch.tensor(0.03),
        torch.tensor(0.38),
        torch.tensor(0.36),
    ]


    lbs = [torch.tensor(i * 0.5) for i in trues[:1]] + [
        torch.tensor(i * (1 - sf)) for i in trues[1:]
    ]
    ubs = [torch.tensor(i * 1.5) for i in trues[:1]] + [
        torch.tensor(i * (1 + sf)) for i in trues[1:]
    ]

    scale_fns = [scaling.BoundedScalingFunction(l, u) for l, u in zip(lbs, ubs)]

    print("Initial parameter values:")
    for n, ic, sfn in zip(names, ics, scale_fns[1:]):
        print("%s:\t%3.2f" % (n, ic), sfn.scale(ic))
    print("")

    actual_maker = lambda *x, **kwargs: make(
        *x, scale_fns, use_adjoint=True, miter=10, **kwargs
    )

    # 3) Create the actual model
    model = optimize.DeterministicModel(actual_maker, names, ics)

    with torch.no_grad():
        pred = model(data, cycles, types, control)
    
    for i in range(pred.shape[-1]):
        plt.plot(data[2, :, i].numpy(), pred[:, i].numpy(), "o", markevery=7, markersize=10, label="Prediction")
        plt.plot(data[2, :, i].numpy(), results[:, i].numpy(), "-", lw=4, label="Actual")
        ax = plt.gca()
        plt.xlabel("Strain", fontsize=23)
        plt.ylabel("Stress (MPa)", fontsize=23)
        plt.legend(loc="best", ncol=1, prop={"size": 20}, frameon=False)
        plt.tick_params(axis="both", which="major", labelsize=23)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("powerlaw-viualize-{}.pdf".format(i))
        # plt.show()
        plt.close()
        
    sys.exit("stop")

    # 4) Setup the optimizer
    niter = 200
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
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
        for name, n in zip(names, model.parameters()):
            print("%s:" % (name), n.data, n.grad)

        end_time = time.time()
        use_time = end_time - start_time
        time_list.append(use_time)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

    # 8) Plot the convergence history
    np.savetxt("loss-newton.txt", loss_history)
    np.savetxt("time-newton.txt", time_list)
