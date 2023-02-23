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

from maker import make_model, load_subset_data

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
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)

def smooth_time(data):
    new_data = torch.empty_like(data)
    for i in range(data.shape[-1]):
        new_data[0, :, i] = torch.linspace(0, data[0, -1, i], data.shape[1])
    new_data[1] = data[1]
    new_data[2] = data[2]
    return new_data


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.00
    nsamples = 1  # 20 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "test-scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )

    data = data[:, :, 0].unsqueeze(-1)
    results = results[:, 0].unsqueeze(-1)
    cycles = cycles[:, 0].unsqueeze(-1)
    types = types[0].unsqueeze(-1)
    control = control[0].unsqueeze(-1)
    data = smooth_time(data)

    
    plt.plot(data[2].numpy(), results.numpy())
    plt.show()
    
    
    sys.exit("stop")
    """
    # visualize
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    ics = [torch.tensor(ra.uniform(0.25, 0.75), device=device) for i in range(5)] + [
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
    ]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
    print("")

    model = optimize.DeterministicModel(make, names, ics)
    loss = torch.nn.MSELoss(reduction="sum")
    lossv = loss(model(data, cycles, types, control), results)
    lossv.backward()
    print("Initial parameter gradients:")
    grads = []
    for n in names:
        grads.append(getattr(model,n).grad.detach())
        print("%s:" % (n), grads[-1])
    print("")

    with torch.no_grad():
        pred = model(data, cycles, types, control)
    plt.plot(data[2].numpy(), pred.numpy())
    plt.plot(data[2].numpy(), results.numpy())
    plt.show()
    plt.close()
    sys.exit("stop")
    """
    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    ics = [torch.tensor(ra.uniform(0.25, 0.75), device=device) for i in range(5)] + [
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
    ]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

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
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(0, 1)
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
        for name, n in zip(names, model.parameters()):
            print("%s:" % (name), n.data, n.grad)

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy (target values are all 0.5):")
    for n in names:
        print(("%s:\t" % n) + str(getattr(model, n).data))

    # 8) Plot the convergence history
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("history.pdf")
    # plt.show()
    plt.close()

    # 9) Visualize
    pred = model(data, cycles, types, control)
    for i in range(data.shape[-1]):
        plt.plot(
            data[2, :, i].detach().numpy(),
            pred[:, i].detach().numpy(),
            label="Prediction",
        )
        plt.plot(
            data[2, :, i].detach().numpy(),
            results[:, i].detach().numpy(),
            "--",
            label="True",
        )
        plt.legend()
        plt.xlabel("Strain")
        plt.ylabel("Stress (MPa)")
        plt.tight_layout()
        plt.savefig("flow-{}.pdf".format(i))
        # plt.show()
        plt.close()
