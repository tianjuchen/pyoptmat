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
from maker import make_model, downsample
from pyoptmat import optimize, experiments, utility, scaling
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
device = torch.device("cpu")

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0


def pgrid(p, T, Q=-100.0):
    return p * torch.exp(-Q / T)


def scale_converter(scales, ics):
    act_ics = [sfn.scale(ic) for ic, sfn in zip(ics, scales)]
    return act_ics


def visualize_variance(strain, stress_true, stress_calc, nsamples, alpha=0.05):
    """
    Visualize variance for batched examples

    Args:
      strain (torch.tensor):        input strain
      stress_true (torch.tensor):   actual stress values
      stress_calc (torch.tensor):   simulated stress values

    Keyword Args:
      alpha (float): alpha value for shading
    """

    ax = plt.gca()
    plt.plot(strain.numpy(), stress_true.numpy(), "-", lw=3)

    plt.plot(strain.numpy(), stress_calc.numpy(), "k-", lw=5)

    plt.xlabel("Strain (mm/mm)", fontsize=18)
    plt.ylabel("Stress (MPa)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    # increase tick width
    ax.tick_params(width=3)
    plt.savefig("batch-stress-strain-{}.pdf".format(nsamples))
    plt.show()
    plt.close()
    return stress_true


def writeout(names, ics):
    for n, i in zip(names, ics):
        print("%s" % (n), "torch." + str(i))
    sys.exit("stop")
    return ics


def optim_params(scale=0.00):
    if scale == 0.0:
        # scale = 0.0
        optim1 = [
            torch.tensor([0.3487, 0.6193, 0.3820, 0.4590, 0.4978]),
            torch.tensor([0.4581, 0.2463, 0.4809, 0.4940, 0.8523]),
            torch.tensor([0.9459, 0.8506, 0.8079, 0.5985, 0.9003]),
            torch.tensor([0.5007, 0.5923, 0.5004, 0.5006, 0.7601]),
            torch.tensor([0.5018, 0.3495, 0.5014, 0.5027, 0.5653]),
        ]

        optim2 = [
            torch.tensor([0.5519, 0.3523, 0.3786, 0.3726, 0.7899]),
            torch.tensor([0.5180, 0.0921, 0.4803, 0.4848, 0.1625]),
            torch.tensor([0.3312, 0.4051, 0.8179, 0.8214, 0.6210]),
            torch.tensor([0.5019, 0.0225, 0.5004, 0.5003, 0.6894]),
            torch.tensor([0.5052, 0.5926, 0.5012, 0.5010, 0.7357]),
        ]

        optim3 = [
            torch.tensor([0.4487, 0.3853, 0.4672, 0.4056, 0.2008]),
            torch.tensor([0.4849, 0.7104, 0.4941, 0.4875, 0.8136]),
            torch.tensor([0.6434, 0.7993, 0.5798, 0.7364, 0.5398]),
            torch.tensor([0.5013, 0.5780, 0.5008, 0.5004, 0.4957]),
            torch.tensor([0.5036, 0.0632, 0.5031, 0.5016, 0.1864]),
        ]

        optim4 = [
            torch.tensor([0.4821, 0.7102, 0.5406, 0.4595, 0.1920]),
            torch.tensor([0.4951, 0.1984, 0.5096, 0.4941, 0.0986]),
            torch.tensor([0.5422, 0.8075, 0.3822, 0.5971, 0.5868]),
            torch.tensor([0.5015, 0.7958, 0.5011, 0.5006, 0.0249]),
            torch.tensor([0.5041, 0.1413, 0.5045, 0.5027, 0.5773]),
        ]

        optim5 = [
            torch.tensor([0.3509, 0.1642, 0.4077, 0.4422, 0.6514]),
            torch.tensor([0.4586, 0.2244, 0.4841, 0.4917, 0.3245]),
            torch.tensor([0.9396, 0.8151, 0.7398, 0.6418, 0.4347]),
            torch.tensor([0.5007, 0.8029, 0.5005, 0.5005, 0.3256]),
            torch.tensor([0.5018, 0.7030, 0.5018, 0.5023, 0.6152]),
        ]
    elif scale == 0.15:
        # scale = 0.15
        optim1 = [
            torch.tensor([0.4783, 0.8260, 0.4779, 0.5929, 0.6831]),
            torch.tensor([0.4902, 0.5084, 0.5589, 0.4747, 0.8717]),
            torch.tensor([0.5912, 0.7460, 0.4352, 0.2696, 0.2822]),
            torch.tensor([0.5260, 0.2252, 0.5260, 0.5196, 0.9269]),
            torch.tensor([0.6078, 0.0264, 0.5434, 0.5240, 0.5015]),
        ]

        optim2 = [
            torch.tensor([0.4739, 0.1205, 0.3220, 0.3728, 0.6139]),
            torch.tensor([0.4888, 0.6880, 0.5370, 0.4419, 0.2505]),
            torch.tensor([0.6044, 0.6068, 0.8796, 0.8153, 0.8631]),
            torch.tensor([0.5260, 0.5386, 0.5253, 0.5185, 0.3373]),
            torch.tensor([0.6078, 0.8022, 0.5405, 0.5185, 0.0578]),
        ]

        optim3 = [
            torch.tensor([0.3998, 0.1780, 0.5498, 0.5948, 0.2780]),
            torch.tensor([0.4673, 0.0230, 0.5754, 0.4752, 0.2980]),
            torch.tensor([0.8282, 0.3405, 0.2305, 0.2639, 0.9095]),
            torch.tensor([0.5255, 0.5468, 0.5263, 0.5195, 0.6005]),
            torch.tensor([0.6064, 0.9198, 0.5446, 0.5239, 0.7019]),
        ]

        optim4 = [
            torch.tensor([0.4406, 0.4911, 0.4740, 0.6599, 0.4510]),
            torch.tensor([0.4788, 0.3404, 0.5581, 0.4898, 0.0290]),
            torch.tensor([0.7049, 0.6077, 0.4464, 0.1027, 0.6579]),
            torch.tensor([0.5258, 0.4792, 0.5260, 0.5198, 0.0125]),
            torch.tensor([0.6072, 0.2648, 0.5433, 0.5253, 0.4832]),
        ]

        optim5 = [
            torch.tensor([0.4669, 0.1221, 0.4415, 0.4871, 0.1029]),
            torch.tensor([0.4867, 0.3722, 0.5518, 0.4550, 0.2655]),
            torch.tensor([0.6253, 0.9715, 0.5392, 0.5313, 0.6897]),
            torch.tensor([0.5260, 0.5974, 0.5259, 0.5191, 0.9303]),
            torch.tensor([0.6077, 0.8199, 0.5428, 0.5213, 0.9682]),
        ]
    return optim1, optim2, optim3, optim4, optim5


def get_ics(optim1, optim2, optim3, optim4, optim5):
    # avg_optim = torch.empty((5, 5))
    avg_optim = []
    for i, (a, b, c, d, e) in enumerate(zip(optim1, optim2, optim3, optim4, optim5)):
        optim = torch.vstack((a, b, c, d, e))
        # avg_optim[i, :] = optim.mean(0)
        avg_optim.append(optim.mean(0))
    print(avg_optim)

    ics = scale_converter(scale_fn, avg_optim)
    return ics


if __name__ == "__main__":

    fnames = ["n", "eta", "s0", "R", "d"]
    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    Tcontrol = torch.tensor([25.0, 100.0, 300.0, 500.0, 600.0]) + 273.15
    Tsmooth = torch.linspace(25.0 + 273.15, 600.0 + 273.15, 100)

    trues = [n_true, eta_true, s0_true, R_true, d_true]
    scale_fn = [
        scaling.BoundedScalingFunction(
            pgrid(p, Tcontrol) * 0.5, pgrid(p, Tcontrol) * 1.5
        )
        for p in trues
    ]
    
    ics_00 = get_ics(*optim_params(scale=0.00))
    ics_15 = get_ics(*optim_params(scale=0.15))

    indices = torch.tensor([0, 2, 3])

    for i, (n, preal) in enumerate(zip(names, trues)):
        ax = plt.gca()
        plt.plot(
            Tsmooth.numpy() - 273.15,
            pgrid(preal, Tsmooth).numpy(),
            "-",
            lw=5,
            label="Arrhenius",
        )
        plt.plot(
            Tcontrol.index_select(0, indices).numpy() - 273.15,
            ics_00[i].index_select(0, indices).numpy(),
            "o",
            markersize=13,
            label="piecewise" + r" $\sigma^{2}=0.00$",
        )
        plt.plot(
            Tcontrol.index_select(0, indices).numpy() - 273.15,
            ics_15[i].index_select(0, indices).numpy(),
            "o",
            markersize=13,
            label="piecewise" + r" $\sigma^{2}=0.15$",
        )
        
        # plt.xlim(20, 600)
        plt.xlabel("Temperature ($^\circ$C)", fontsize=23)
        plt.ylabel(n, fontsize=30)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        plt.locator_params(axis="both", nbins=4)
        if i == len(names) - 5:
            plt.legend(
                loc="best",
                frameon=True,
                prop={"size": 18},
            )

        plt.tight_layout()
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3)
        # increase tick width
        ax.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("scatter-tfn-optim-{}.pdf".format(fnames[i]))
        # plt.show()
        plt.close()
