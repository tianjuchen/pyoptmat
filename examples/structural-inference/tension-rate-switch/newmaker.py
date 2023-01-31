import sys

sys.path.append("../../../..")
import xarray as xr
from pyoptmat import (
    optimize,
    experiments,
    temperature,
    hardening,
    models,
    flowrules,
    utility,
)
from pyoptmat.temperature import ConstantParameter as CP
import torch
import tqdm
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.easyguide import easy_guide
import pyro.optim as optim
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import argparse
from pyro import poutine
import pyro.distributions as dist
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# Run on CPU (home machine GPU is eh)
dev = "cpu"
device = torch.device(dev)

class extrapolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value(self, T):
        func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
        return torch.tensor(func(T).tolist(), device=self.xs.device)


weights_input = {
    "tensile": 1.0,
    "relaxation": 1.0,
    "strain_cyclic": 1.0,
    "creep": 1.0,
    "stress_cyclic": 1.0,
    "abstract_tensile": 1.0,
    "direct_data": 1.0,
}


def construct_weights(etypes, weights, normalize=True):
    """
    Construct an array of weights

    Parameters:
      etypes:       strings giving the experiment type
      weights:      dictionary mapping etype: weight

    Additional Parameters:
      normalize:    normalize by the number of experiments of each type
    """
    warray = torch.ones(len(etypes))

    count = defaultdict(int)
    for i, et in enumerate(etypes):
        warray[i] = weights[et]
        count[et] += 1

    if normalize:
        for i, et in enumerate(etypes):
            warray[i] /= count[et]

    return warray


def data_extraction_type(
    exp_data, exp_results, exp_cycles, exp_types, exp_control, etype, T_min, T_max
):
    relevant = torch.logical_and(
        exp_types != experiments.exp_map[etype],
        torch.logical_and(exp_data[1, 0] < T_max, exp_data[1, 0] > T_min),
    )
    data = exp_data[:, :, relevant]
    cycles = exp_cycles[:, relevant]
    types = exp_types[relevant]
    control = exp_control[relevant]
    results = exp_results[:, relevant]
    return data, results, cycles, types, control


def data_extraction_control(
    exp_data, exp_results, exp_cycles, exp_types, exp_control, econtrol, T_min, T_max
):
    relevant = torch.logical_and(
        exp_control == experiments.control_map[econtrol],
        torch.logical_and(exp_data[1, 0] < T_max, exp_data[1, 0] > T_min),
    )
    data = exp_data[:, :, relevant]
    cycles = exp_cycles[:, relevant]
    types = exp_types[relevant]
    control = exp_control[relevant]
    results = exp_results[:, relevant]
    return data, results, cycles, types, control

def data_extraction_temperature(
    exp_data, exp_results, exp_cycles, exp_types, exp_control, T_min, T_max
):
    relevant = torch.logical_and(exp_data[1, 0] < T_max, exp_data[1, 0] > T_min)
    data = exp_data[:, :, relevant]
    cycles = exp_cycles[:, relevant]
    types = exp_types[relevant]
    control = exp_control[relevant]
    results = exp_results[:, relevant]
    return data, results, cycles, types, control


def convert_value_to_key(etypes, dictionary):
    type_keys = []
    for key, value in dictionary.items():
        for j in etypes:
            if j == value:
                type_keys.append(key)

    return type_keys


def sample_from_prior(sampling=False):

    if sampling:
        n_mean = torch.tensor(
            [
                4.1833,
                4.2726,
                3.6977,
                3.6665,
                3.5967,
                4.2726,
                3.4216,
                4.2726,
                3.3626,
                4.2765,
            ]
        )
        n_std = torch.tensor(
            [
                0.0031,
                0.0025,
                0.0129,
                0.0028,
                0.0024,
                0.0025,
                0.0021,
                0.0025,
                0.0114,
                0.0023,
            ]
        )
        eta_mean = torch.tensor(
            [
                2102.5446,
                2017.3121,
                1858.4626,
                1798.1494,
                1800.8548,
                2017.3121,
                1770.9661,
                2017.3121,
                1666.9144,
                2017.8883,
            ]
        )
        eta_std = torch.tensor(
            [
                1.6159,
                1.0294,
                4.6617,
                1.2840,
                1.1199,
                1.0294,
                0.9027,
                1.0294,
                1.7946,
                1.0232,
            ]
        )
        br_mean = torch.tensor(
            [
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                299.1483,
                299.1483,
            ]
        )
        br_std = torch.tensor(
            [
                0.2168,
                0.2168,
                0.2168,
                0.2168,
                0.2168,
                0.2168,
                0.2168,
                0.2168,
                0.4023,
                0.4023,
            ]
        )
        bh_mean = torch.tensor(
            [
                6.1927,
                18.9977,
                4.8496,
                4.5919,
                6.7625,
                18.9977,
                11.4115,
                18.9977,
                33.2229,
                18.9003,
            ]
        )
        bh_std = torch.tensor(
            [
                0.0140,
                0.0238,
                0.0147,
                0.0112,
                0.0146,
                0.0238,
                0.6765,
                0.0238,
                3.1372,
                0.0238,
            ]
        )
        A_mean = torch.tensor(
            [
                -210.0751,
                -126.8421,
                -208.6932,
                -208.9388,
                -216.2307,
                -126.8421,
                -230.6615,
                -126.8421,
                -251.5020,
                -126.8421,
            ]
        )
        A_std = torch.tensor(
            [
                0.1276,
                0.0853,
                0.1207,
                0.1371,
                0.0833,
                0.0853,
                2.1047,
                0.0853,
                1.2653,
                0.0853,
            ]
        )
        B_mean = torch.tensor(
            [
                -140.5268,
                -126.9174,
                -140.3390,
                -139.9922,
                -133.1797,
                -126.9174,
                -116.0600,
                -126.9174,
                -96.8560,
                -126.9174,
            ]
        )
        B_std = torch.tensor(
            [
                0.1379,
                0.0560,
                0.1344,
                0.1424,
                0.1188,
                0.0560,
                1.8820,
                0.0560,
                1.0255,
                0.0560,
            ]
        )
        s0_mean = torch.tensor(6.7498)
        s0_std = torch.tensor([0.8558])

        n = dist.Normal(n_mean, n_std).sample()
        eta = dist.Normal(eta_mean, eta_std).sample()
        br = dist.Normal(br_mean, br_std).sample()
        bh = dist.Normal(bh_mean, bh_std).sample()
        A = dist.Normal(A_mean, A_std).sample()
        B = dist.Normal(B_mean, B_std).sample()
        s0 = dist.Normal(s0_mean, s0_std).sample()
    else:
        n = torch.tensor(
            [
                4.1833,
                4.2726,
                3.6977,
                3.6665,
                3.5967,
                4.2726,
                3.4216,
                4.2726,
                3.3626,
                4.2765,
            ]
        )
        eta = torch.tensor(
            [
                2102.5446,
                2017.3121,
                1858.4626,
                1798.1494,
                1800.8548,
                2017.3121,
                1770.9661,
                2017.3121,
                1666.9144,
                2017.8883,
            ]
        )
        br = torch.tensor(
            [
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                303.1989,
                299.1483,
                299.1483,
            ]
        )
        bh = torch.tensor(
            [
                6.1927,
                18.9977,
                4.8496,
                4.5919,
                6.7625,
                18.9977,
                11.4115,
                18.9977,
                33.2229,
                18.9003,
            ]
        )
        A = torch.tensor(
            [
                -210.0751,
                -126.8421,
                -208.6932,
                -208.9388,
                -216.2307,
                -126.8421,
                -230.6615,
                -126.8421,
                -251.5020,
                -126.8421,
            ]
        )
        B = torch.tensor(
            [
                -140.5268,
                -126.9174,
                -140.3390,
                -139.9922,
                -133.1797,
                -126.9174,
                -116.0600,
                -126.9174,
                -96.8560,
                -126.9174,
            ]
        )
        s0 = torch.tensor(6.7498)

    return n, eta, br, bh, A, B, s0


def new_maker(
    n,
    eta,
    br, 
    bh, 
    A, 
    B, 
    s0,
    C1,
    C2,
    C3,
    g1,
    g2,
    g3,
    b1,
    b2,
    b3,
    d1,
    d2,
    d3,
    alpha,
    beta,
    gamma,
    scale_functions = None,
    **kwargs
):

    device = C1.device
    Ts = Tcontrol = (
        torch.tensor([20.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        + 273.15
    )
    Es = torch.tensor(
        [
            196.5e3,
            191.3e3,
            184.8e3,
            178.3e3,
            171.6e3,
            165.0e3,
            157.7e3,
            150.1e3,
            141.3e3,
        ]
    )

    Tcontrol_new = (
        torch.tensor(
            [
                19.0,
                100.0,
                200.0,
                300.0,
                400.0,
                450.0,
                500.0,
                550.0,
                600.0,
                650.0,
                680.0,
                700.0,
                720.0,
                750.0,
                761.0,
            ]
        )
        + 273.15
    )


    if scale_functions is None:
        n_scale = lambda x: x
        eta_scale = lambda x: x
        br_scale = lambda x: x
        bh_scale = lambda x: x
        A_scale = lambda x: x
        B_scale = lambda x: x
        s0_scale = lambda x: x
        C1_scale = lambda x: x
        C2_scale = lambda x: x
        C3_scale = lambda x: x
        g1_scale = lambda x: x
        g2_scale = lambda x: x
        g3_scale = lambda x: x
        b1_scale = lambda x: x
        b2_scale = lambda x: x
        b3_scale = lambda x: x
        d1_scale = lambda x: x
        d2_scale = lambda x: x
        d3_scale = lambda x: x
        alpha_scale = lambda x: x
        beta_scale = lambda x: x
        gamma_scale = lambda x: x
    else:
        n_scale = scale_functions[0]
        eta_scale = scale_functions[1]
        br_scale = scale_functions[2] 
        bh_scale = scale_functions[3] 
        A_scale = scale_functions[4]
        B_scale = scale_functions[5]
        s0_scale = scale_functions[6]
        C1_scale = scale_functions[7]
        C2_scale = scale_functions[8]
        C3_scale = scale_functions[9]
        g1_scale = scale_functions[10]
        g2_scale = scale_functions[11]
        g3_scale = scale_functions[12]
        b1_scale = scale_functions[13]
        b2_scale = scale_functions[14]
        b3_scale = scale_functions[15]
        d1_scale = scale_functions[16]
        d2_scale = scale_functions[17]
        d3_scale = scale_functions[18]
        alpha_scale = scale_functions[19]
        beta_scale = scale_functions[20]
        gamma_scale = scale_functions[21]

    Evalues = extrapolate(Ts, Es).value(Tcontrol_new)

    iso = hardening.YaguchiHardeningModel(
        temperature.PiecewiseScaling(Tcontrol_new, br, values_scale_fn=br_scale),
        temperature.PiecewiseScaling(Tcontrol_new, bh, values_scale_fn=bh_scale),
        temperature.PiecewiseScaling(Tcontrol_new, A, values_scale_fn=A_scale),
        temperature.PiecewiseScaling(Tcontrol_new, B, values_scale_fn=B_scale),
    )

    kin1 = hardening.FAKinematicHardeningModelRecovery(
        temperature.PiecewiseScaling(Tcontrol_new, C1, values_scale_fn=C1_scale),
        temperature.PiecewiseScaling(Tcontrol_new, g1, values_scale_fn=g1_scale),
        temperature.PiecewiseScaling(Tcontrol_new, b1, values_scale_fn=b1_scale),
        temperature.PiecewiseScaling(Tcontrol_new, d1, values_scale_fn=d1_scale),
    )
    kin2 = hardening.FAKinematicHardeningModelRecovery(
        temperature.PiecewiseScaling(Tcontrol_new, C2, values_scale_fn=C2_scale),
        temperature.PiecewiseScaling(Tcontrol_new, g2, values_scale_fn=g2_scale),
        temperature.PiecewiseScaling(Tcontrol_new, b2, values_scale_fn=b2_scale),
        temperature.PiecewiseScaling(Tcontrol_new, d2, values_scale_fn=d2_scale),
    )
    kin3 = hardening.FAKinematicHardeningModelRecovery(
        temperature.PiecewiseScaling(Tcontrol_new, C3, values_scale_fn=C3_scale),
        temperature.PiecewiseScaling(Tcontrol_new, g3, values_scale_fn=g3_scale),
        temperature.PiecewiseScaling(Tcontrol_new, b3, values_scale_fn=b3_scale),
        temperature.PiecewiseScaling(Tcontrol_new, d3, values_scale_fn=d3_scale),
    )

    kin = hardening.SuperimposedKinematicHardening([kin1, kin2, kin3])

    fr = flowrules.IsoKinPartialViscoplasticity(
        temperature.PiecewiseScaling(Tcontrol_new, n, values_scale_fn=n_scale),
        temperature.PiecewiseScaling(Tcontrol_new, eta, values_scale_fn=eta_scale),
        CP(s0, p_scale=s0_scale),
        temperature.PiecewiseScaling(
            Tcontrol_new, alpha, values_scale_fn=alpha_scale
        ),
        temperature.PiecewiseScaling(Tcontrol_new, beta, values_scale_fn=beta_scale),
        temperature.PiecewiseScaling(
            Tcontrol_new, gamma, values_scale_fn=gamma_scale
        ),
        iso,
        kin,
    )
    model = models.InelasticModel(temperature.PiecewiseScaling(Tcontrol_new, Evalues), fr)
    integrator = models.ModelIntegrator(model, **kwargs)

    return integrator


def index_select(data, Ninterval):

    if data.dim() == 3:
        new_data = torch.empty(
            (data.shape[0],) + (int(data.shape[1] / Ninterval), data.shape[2])
        )
        indices = torch.arange(0, data.shape[1], Ninterval)
        for i in range(data.shape[0]):
            new_data[i] = data[i].index_select(0, indices)
    elif data.dim() == 2:
        indices = torch.arange(0, data.shape[0], Ninterval)
        new_data = torch.index_select(data, 0, indices)
    elif data.dim() == 1:
        new_data = data
    return new_data


def fewer_sample(Ndata, data, results, cycles, types, control):

    new_data = index_select(data, Ndata)
    new_results = index_select(results, Ndata)
    new_cycles = index_select(cycles, Ndata)
    new_types = index_select(types, Ndata)
    new_control = index_select(control, Ndata)

    return new_data, new_results, new_cycles, new_types, new_control




