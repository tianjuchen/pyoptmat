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
from scipy import interpolate

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
            torch.tensor([0.1661, 0.4452, 0.4120, 0.5391, 0.4722]),
            torch.tensor([0.3757, 0.4858, 0.4787, 0.5058, 0.2996]),
            torch.tensor([0.0393, 0.5713, 0.7202, 0.4179, 0.5552]),
            torch.tensor([0.6052, 0.4893, 0.4956, 0.5044, 0.6200]),
            torch.tensor([0.4969, 0.4862, 0.5022, 0.4808, 0.4869]),
        ]

        optim2 = [
            torch.tensor([0.5418, 0.5235, 0.6141, 0.4907, 0.5400]),
            torch.tensor([0.6934, 0.4973, 0.5218, 0.5148, 0.8800]),
            torch.tensor([0.1979, 0.3782, 0.1794, 0.4771, 0.1652]),
            torch.tensor([0.7446, 0.4886, 0.4965, 0.5060, 0.5024]),
            torch.tensor([0.7384, 0.4844, 0.5053, 0.4925, 0.8930]),
        ]

        optim3 = [
            torch.tensor([0.2968, 0.5644, 0.5423, 0.6029, 0.4435]),
            torch.tensor([0.2078, 0.5188, 0.5023, 0.5288, 0.9657]),
            torch.tensor([0.0392, 0.2240, 0.3825, 0.2036, 0.8701]),
            torch.tensor([0.9601, 0.4897, 0.4962, 0.5044, 0.4296]),
            torch.tensor([0.7820, 0.4919, 0.4998, 0.5039, 0.6119]),
        ]

        optim4 = [
            torch.tensor([0.9099, 0.4145, 0.4065, 0.6548, 0.4423]),
            torch.tensor([0.7297, 0.4736, 0.4816, 0.5181, 0.6100]),
            torch.tensor([0.7768, 0.6804, 0.7176, 0.1654, 0.7733]),
            torch.tensor([0.4578, 0.4884, 0.4961, 0.5039, 0.5555]),
            torch.tensor([0.2933, 0.4837, 0.5041, 0.4761, 0.6744]),
        ]

        optim5 = [
            torch.tensor([0.1056, 0.6138, 0.5177, 0.4501, 0.9045]),
            torch.tensor([0.8799, 0.5435, 0.4894, 0.5155, 0.3016]),
            torch.tensor([0.7230, 0.0520, 0.4689, 0.5589, 0.5820]),
            torch.tensor([0.0568, 0.4910, 0.4953, 0.5066, 0.9625]),
            torch.tensor([0.7799, 0.4926, 0.5003, 0.4963, 0.7989]),
        ]
    elif scale == 0.15:
        # scale = 0.15
        optim1 = [
            torch.tensor([0.2187, 0.9, 0.1416, 0.5763, 0.6390]),
            torch.tensor([0.4715, 0.2324, 0.6083, 0.4256, 0.9552]),
            torch.tensor([0.9883, 0.1644, 0.9943, 0.9920, 0.6187]),
            torch.tensor([0.4761, 0.4518, 0.4519, 0.4992, 0.8425]),
            torch.tensor([0.24, 0.1, 0.7, 0.4, 0.12]),
        ]

        optim2 = [
            torch.tensor([0.1360, 0.9978, 0.1412, 0.5750, 0.5566]),
            torch.tensor([0.6894, 0.2362, 0.6076, 0.4261, 0.7748]),
            torch.tensor([0.1370, 0.1458, 0.9984, 0.9927, 0.5677]),
            torch.tensor([0.2456, 0.4531, 0.4516, 0.4995, 0.0923]),
            torch.tensor([0.68, 0.1, 0.7, 0.4, 0.13]),
        ]

        optim3 = [
            torch.tensor([0.8910, 0.9991, 0.1410, 0.5768, 0.6640]),
            torch.tensor([0.8665, 0.2343, 0.6076, 0.4255, 0.8183]),
            torch.tensor([0.3725, 0.1533, 0.9977, 0.9926, 0.0320]),
            torch.tensor([0.7195, 0.4518, 0.4517, 0.4992, 0.0597]),
            torch.tensor([0.47, 0.1, 0.73, 0.40, 0.1]),
        ]

        optim4 = [
            torch.tensor([0.4270, 0.9, 0.1407, 0.5749, 0.6318]),
            torch.tensor([0.4906, 0.2375, 0.6071, 0.4261, 0.2135]),
            torch.tensor([0.6605, 0.1380, 1.0000, 0.9934, 0.8764]),
            torch.tensor([0.6759, 0.4532, 0.4513, 0.4996, 0.6083]),
            torch.tensor([0.5728, 0.1, 0.7372, 0.3985, 0.1152]),
        ]

        optim5 = [
            torch.tensor([0.8328, 0.9993, 0.1417, 0.5756, 0.4799]),
            torch.tensor([0.0511, 0.2365, 0.6075, 0.4258, 0.6027]),
            torch.tensor([0.5733, 0.1437, 0.9964, 0.9933, 0.9726]),
            torch.tensor([0.0545, 0.4528, 0.4516, 0.4994, 0.0543]),
            torch.tensor([0.9, 0.1, 0.7, 0.4, 0.1]),
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


# def interp(x, y, x_new):
    # return interpolate.interp1d(x, y)(x_new)


def interp(x, y, x_new):
    return np.interp(x_new, x, y)


if __name__ == "__main__":

    fnames = ["n", "eta", "s0", "R", "d"]
    names = [r"$n$", r"$\eta$", r"$s_{0}$", r"$R$", r"$d$"]
    Tcontrol = torch.tensor([25.0, 100.0, 300.0, 500.0, 600.0]) + 273.15
    Tsmooth = torch.linspace(25.0 + 273.15, 600.0 + 273.15, 100)
    Tcontrol_new = torch.tensor(
        [415.1500, 485.1500, 503.1500, 511.1500, 598.1500, 600.1500, 615.1500, 758.1500]
    )

    trues = [n_true, eta_true, s0_true, R_true, d_true]
    scale_fn = [
        scaling.BoundedScalingFunction(
            pgrid(p, Tcontrol) * 0.5, pgrid(p, Tcontrol) * 1.5
        )
        for p in trues
    ]

    ics_00 = get_ics(*optim_params(scale=0.00))
    ics_15 = get_ics(*optim_params(scale=0.15))

    # indices = torch.tensor([0, 1, 2, 3, 4])

    for i, (n, preal) in enumerate(zip(names, trues)):
        ax = plt.gca()
        plt.plot(
            Tsmooth.numpy() - 273.15,
            pgrid(preal, Tsmooth).numpy(),
            "-",
            lw=5,
            label="Arrhenius",
        )
        """
        plt.plot(
        Tcontrol.numpy() - 273.15,
        ics_00[i].numpy(),
        "o",
        markersize=13,
        label="piecewise" + r" $\sigma^{2}=0.00$",
        )
        
        plt.plot(
            Tcontrol.numpy() - 273.15,
            ics_15[i].numpy(),
            "o",
            markersize=13,
            label="piecewise" + r" $\sigma^{2}=0.15$",
        )
        
        """
        res_00 = interp(
            Tcontrol.numpy(),
            ics_00[i].numpy(),
            Tcontrol_new.numpy(),
        )

        plt.plot(
        Tcontrol_new.numpy() - 273.15,
        res_00,
        "o",
        markersize=13,
        label="piecewise" + r" $\sigma^{2}=0.00$",
        )

        res_15 = interp(
            Tcontrol.numpy(),
            ics_15[i].numpy(),
            Tcontrol_new.numpy(),
        )

        plt.plot(
            Tcontrol_new.numpy() - 273.15,
            res_15,
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
        plt.savefig("scatter-tfn-develop-{}.pdf".format(fnames[i]))
        # plt.show()
        plt.close()
