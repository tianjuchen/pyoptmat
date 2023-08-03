#!/usr/bin/env python3

import sys

sys.path.append("../../..")
import xarray as xr
import optimize as opt
from pyoptmat import (
    optimize,
    experiments,
    temperature,
    hardening,
    models,
    flowrules,
    utility,
    scaling,
)
import process
import maker, ics
from pyoptmat.temperature import ConstantParameter as CP
import torch
import tqdm
import pandas as pd
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
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import RcParams
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


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)


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


def plot_prop(exp_type, exp_control, etype, T, index_number=None, fs=23):
    plt.legend(
        [
            Line2D([0], [0], color="k", ls="--"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="lightskyblue", edgecolor=None, alpha=0.5),
        ],
        ["Experimental data", "Model average", r"Model 95$\%$ prediction interval"],
        loc="best",
        prop={"size": 14},
    )

    if (exp_type == 2) or (exp_type == 4):
        plt.xlabel("Cycles (N)", fontsize=fs)
    elif exp_type == 3:
        plt.xlabel("Time (s)", fontsize=fs)
    else:
        plt.xlabel("Strain (mm/mm)", fontsize=fs)

    if exp_control == 0:
        plt.ylabel("Stress (MPa)", fontsize=fs)
    else:
        plt.ylabel("Strain (mm/mm)", fontsize=fs)

    plt.tick_params(axis="both", which="major", labelsize=fs)
    plt.tight_layout()
    if index_number is not None:
        plt.savefig(
            "gr91-{}_%i-C-{}.pdf".format(etype, index_number) % int(T - 273.15)
        )
    else:
        plt.savefig("gr91-{}_%i-C.pdf".format(etype) % int(T - 273.15))
    # plt.show()
    plt.close()


def fecthdata(
    exp_data,
    exp_results,
    exp_cycles,
    exp_types,
    exp_control,
    etype,
    T,
    dT=2.0,
):
    relevant = torch.logical_and(
        exp_types == experiments.exp_map[etype],
        torch.logical_and(exp_data[1, 0] < T + dT, exp_data[1, 0] > T - dT),
    )
    data = exp_data[:, :, relevant]
    cycles = exp_cycles[:, relevant]
    types = exp_types[relevant]
    control = exp_control[relevant]
    results = exp_results[:, relevant]

    return data, results, cycles, types, control


def stressdata(
    exp_data,
    exp_results,
    exp_cycles,
    exp_types,
    exp_control,
    sig,
    dsig=2.0,
):
    relevant = torch.logical_and(
        exp_data[2, -1] < sig + dsig, exp_data[2, -1] > sig - dsig
    )
    data = exp_data[:, :, relevant]
    cycles = exp_cycles[:, relevant]
    types = exp_types[relevant]
    control = exp_control[relevant]
    results = exp_results[:, relevant]

    return data, results, cycles, types, control


def uq_visualize(
    sig,
    T,
    model,
    nsamples,
    alpha,
    data,
    results,
    cycles,
    types,
    control,
    acord_load=False,
):
    stress_results = torch.zeros(nsamples, data.shape[1], device=device)
    for i in trange(nsamples):
        with torch.no_grad():
            stress_results[i, :] = model(data, cycles, types, control)[:, 0]

    mean_result = torch.mean(stress_results, 0)
    sresults, _ = torch.sort(stress_results, 0)
    min_result = sresults[int(alpha * nsamples), :]
    max_result = sresults[int((1 - alpha) * nsamples), :]
    plt.style.use(latex_style_times)
    plt.figure()
    if (types[0] == 2) or (types[0] == 4):
        every_single_plot = False
        if every_single_plot:
            for j in trange(data.shape[-1]):
                plt.plot(cycles[:, j].cpu(), results[:, j].cpu(), "k--", lw=4)
                (l,) = plt.plot(cycles[:, j].cpu(), mean_result, lw=4, color="k")
                p = plt.fill_between(
                    cycles[:, j].cpu(),
                    min_result,
                    max_result,
                    alpha=0.5,
                    color="lightskyblue",
                )
                plot_prop(types[0], control[0], etype, T, index_number=j)
        else:
            plt.plot(
                cycles[:, :nsamples].cpu(), results[:, :nsamples].cpu(), "k--", lw=4
            )
            (l,) = plt.plot(cycles[:, 0].cpu(), mean_result, lw=4, color="k")
            p = plt.fill_between(
                cycles[:, 0].cpu(),
                min_result,
                max_result,
                alpha=0.5,
                color="lightskyblue",
            )
            plot_prop(types[0], control[0], etype, T)

    elif types[0] == 3:
        print("start plotting each creeping")
        plt.plot(data[0, :, :nsamples].cpu(), results[:, :nsamples].cpu(), "k--", lw=4)
        (l,) = plt.plot(data[0, :, 0], mean_result, lw=4, color="k")
        p = plt.fill_between(
            data[0, :, 0],
            min_result,
            max_result,
            alpha=0.5,
            color="lightskyblue",
        )
        if acord_load:
            plot_prop(types[0], control[0], etype, T, index_number=sig)
        else:
            plot_prop(types[0], control[0], etype, T)

    else:
        plt.plot(data[2, :, :nsamples].cpu(), results[:, :nsamples].cpu(), "k--", lw=4)
        (l,) = plt.plot(data[2, :, 0], mean_result, lw=4, color="k")
        p = plt.fill_between(
            data[2, :, 0], min_result, max_result, alpha=0.5, color="lightskyblue"
        )
        plot_prop(types[0], control[0], etype, T)

    return min_result, max_result, mean_result


def save_list(data, min_result, max_result, mean_result, etype, sig, T):
    # save the stress-strain data for future use
    pred_res = pd.DataFrame(
        {
            "min": min_result.detach().numpy(),
            "max": max_result.detach().numpy(),
            "mean": mean_result.detach().numpy(),
        }
    )
    pred_res.to_csv("{}_%i-C-{}.csv".format(etype, int(sig)) % int(T - 273.15))

    return pred_res


def scale_converter(scales, ics):
    act_ics = [sfn.scale(ic) for ic, sfn in zip(ics, scales)]
    return act_ics


def unscale_converter(scales, ics):
    act_ics = [sfn.unscale(ic) for ic, sfn in zip(ics, scales)]
    return act_ics


def posteriors_preview(
    names,
    scales,
    real_initial_values,
    loc_loc_prior,
    scale_scale_prior,
    lbs,
    ubs,
    scaling,
):
    if scaling:
        print("Posterior distributions:")
        for name, sfn, iv, loc, scale, lb, ub in zip(
            names,
            scales,
            real_initial_values,
            loc_loc_prior,
            scale_scale_prior,
            lbs,
            ubs,
        ):
            print(name)

            print("\tUnnormalize")
            print("\tloc:", sfn(loc))
            dinc = ub - lb
            print("\t\tscale:", scale * dinc)

            print("\tNormalized")
            print("\t\tloc:", loc)
            print("\t\tscale:", scale)

        print("")
    else:
        print("Posterior distributions:")
        for name, loc, scale in zip(names, loc_loc_prior, scale_scale_prior):
            print(name)
            print("\tloc:", loc)
            print("\tscale:", scale)

    return names


def predict(
    f,
    names,
    Tcontrol,
    real_initial_values,
    variability,
    scales,
    lbs,
    ubs,
    eps,
    model_maker,
    nsamples,
    alpha,
    etype,
    T_min=15.0 + 273.15,
    T_max=762.0 + 273.15,
    time_chunk_size=40,
    pscaling=True,
    acord_load=False,
):
    exp_data, exp_results, exp_cycles, exp_types, exp_control = opt.load_data(f, device)
    print(exp_data.shape)
    print("type is:", maker.convert_value_to_key(exp_types, experiments.exp_map))

    # Scaling functions
    if pscaling:
        real_ics = scale_converter(scales, real_initial_values)
        loc_loc_prior = unscale_converter(scales, real_ics)
        scale_scale_prior = variability
        _ = posteriors_preview(
            names,
            scales,
            real_ics,
            loc_loc_prior,
            scale_scale_prior,
            lbs,
            ubs,
            pscaling,
        )
    else:
        loc_loc_prior = real_initial_values
        scales = None
        scale_scale_prior = variability
        _ = posteriors_preview(
            names,
            scales,
            real_initial_values,
            loc_loc_prior,
            scale_scale_prior,
            lbs,
            ubs,
            pscaling,
        )
    # Add some options
    actual_maker = lambda *x, **kwargs: model_maker(
        *x,
        scale_functions=scales,
        Tcontrol=Tcontrol,
        #block_size = time_chunk_size,
        use_adjoint=True,
        miter=10,
        **kwargs
    )

    # Setup model, guide, and param_store
    model = optimize.StatisticalModel(
        actual_maker, names, loc_loc_prior, scale_scale_prior, eps
    ).to(device)

    # fectch data based on temperature level
    mode = exp_types == experiments.exp_map[etype]
    indice = (mode == True).nonzero(as_tuple=True)[0]
    temperatures = exp_data[1]
    temperatures = temperatures.index_select(1, indice)
    temperature_lists = torch.unique(temperatures[0, :])

    # extract temperature based on the temperature range
    conditions = torch.logical_and(temperature_lists < T_max, temperature_lists > T_min)
    temperature_lists = temperature_lists[conditions]
    print("temperature: ", temperature_lists)

    if etype == "creep":
        for T in temperature_lists:
            print("current temperature is:", T)
            use_data, use_results, use_cycles, use_types, use_control = fecthdata(
                exp_data,
                exp_results,
                exp_cycles,
                exp_types,
                exp_control,
                etype,
                T,
            )
            load_list = use_data[2, -1, :].unique()#[None, ...]
            if acord_load:
                for sig in load_list:
                    print("current load is:", sig)
                    data, results, cycles, types, control = stressdata(
                        use_data,
                        use_results,
                        use_cycles,
                        use_types,
                        use_control,
                        sig,
                    )
                    print(
                        data.size(),
                        cycles.size(),
                        types.size(),
                        control.size(),
                        results.size(),
                    )
                    
                    min_result, max_result, mean_result = uq_visualize(
                        sig,
                        T,
                        model,
                        nsamples,
                        alpha,
                        data,
                        results,
                        cycles,
                        types,
                        control,
                        acord_load,
                    )
                    _ = save_list(
                        data, min_result, max_result, mean_result, etype, sig, T
                    )
            else:
                data, results, cycles, types, control = (
                    use_data,
                    use_results,
                    use_cycles,
                    use_types,
                    use_control,
                )

                print(
                    data.size(),
                    cycles.size(),
                    types.size(),
                    control.size(),
                    results.size(),
                )
                min_result, max_result, mean_result = uq_visualize(
                    0.0,
                    T,
                    model,
                    nsamples,
                    alpha,
                    data,
                    results,
                    cycles,
                    types,
                    control,
                    acord_load,
                )
    else:
        for T in temperature_lists:
            data, results, cycles, types, control = fecthdata(
                exp_data,
                exp_results,
                exp_cycles,
                exp_types,
                exp_control,
                etype,
                T,
            )
            print("current type is:", maker.convert_value_to_key(types, experiments.exp_map))
            print(
                data.size(), cycles.size(), types.size(), control.size(), results.size()
            )
            min_result, max_result, mean_result = uq_visualize(
                0.0, T, model, nsamples, alpha, data, results, cycles, types, control
            )

    return model


def scale_producer(names, lbs, ubs, pscaling=True):
    if pscaling:
        scales = []
        for name, l, u in zip(names, lbs, ubs):
            if name == "r1":
                sfn = scaling.LogBoundedScalingFunction(l, u)
                scales.append(sfn)
            else:
                sfn = scaling.BoundedScalingFunction(l, u)
                scales.append(sfn)
    else:
        scales = None
    return scales


def tgrid(T, ics, device):
    new_grid = (
        torch.tensor(
            [14.7119, 56.5000, 421.5000, 494.5000, 567.5000, 640.5000, 762.0342],
            device=device,
        )
        + 273.15
    )
    res = []
    for i in ics:
        data = inter.interp1d(T.cpu().clone().numpy(), i.cpu().numpy())(
            new_grid.cpu().clone().numpy()
        )
        res.append(torch.tensor(data, device=device))
    return res, new_grid


def gr91_ics(device):
    T = torch.tensor([19.0, 200.0, 500.0, 600.0, 700.0, 761.0], device=device) + 273.15
    mean = [
        torch.tensor([105819.4240, 153387.6150, 148085.8644, 123366.2677, 113714.1560, 120691.5352]),
        torch.tensor([17.7833, 23.0727, 13.4889, 11.6325, 7.1498, 6.5642]),
        torch.tensor([782.6721, 760.9123, 843.0380, 426.5559, 344.8394, 326.7267]),
        torch.tensor([17.2876, 37.1764, 16.9134, 29.4142, 31.4223, 28.4437]),
        torch.tensor([76.5573, 475.5531, 322.4190, 57.8291, 28.6868, 27.9341]),
        torch.tensor([1959.7094, 6881.5758, 1084.6596, 3656.4953, 1333.4722, 1188.5170]),
        torch.tensor([81.9564, 130.7876, 0.1438, 3.8015, 12.5429, 15.1714]),
        torch.tensor([7.3845e-11, 8.8561e-11, 4.2630e-11, 2.3048e-11, 7.9581e-12, 6.3539e-12]),
        torch.tensor([3.8562, 4.5071, 2.5772, 2.4613, 3.1038, 1.8780]),
    ]

    std = [
        torch.tensor([5548.0784, 11342.8433, 10437.4927, 6887.1651, 5442.5053, 6885.8866]),
        torch.tensor([0.3380, 1.8749, 2.1060, 1.4329, 0.4126, 0.1276]),
        torch.tensor([9.3893, 65.6771, 131.4406, 46.1369, 19.7154, 2.7284]),
        torch.tensor([0.3668, 3.1336, 2.6811, 3.8161, 1.9106, 0.4169]),
        torch.tensor([2.8108, 60.1768, 73.4156, 13.8224, 5.2399, 1.4133]),
        torch.tensor([54.0402, 840.6218, 277.2022, 1100.4323, 289.3557, 155.1924]),
        torch.tensor([0.7542, 22.0446, 0.0239, 2.8940, 3.2158, 8.3156]),
        torch.tensor([1.4574e-12, 1.7737e-11, 1.1490e-11, 1.5901e-11, 3.4896e-12, 1.7832e-12]),
        torch.tensor([0.0875, 0.7255, 0.5728, 1.8998, 2.3022, 1.4242]),
    ]     
    return T, mean, std


if __name__ == "__main__":
    mats = ["ss316", "ss304", "gr91", "a800", "a617"]
    mat = mats[2]
    print("current mat is:", mat)
    f = mat + ".nc"

    # define input
    nsamples = 25
    confidence = 0.05 / 2
    etypes = ["tensile", "creep", "strain_cyclic"]
    model_maker = maker.chaboche_maker

    Tcontrol, _, _ = gr91_ics(device)
    #scalefns = ics.mono_scale_fns(Tcontrol, device)
    #names = scalefns.keys()

    names = [
        "C1",
        "C2",
        "C3",
        "g1",
        "g2",
        "g3",
        "b1",
        "b2",
        "b3",
        "d1",
        "d2",
        "d3",
    ]

    posteriors = [
        torch.tensor([-207.9447, -207.9815, -525.5487, -489.8230, -196.4843, -189.1120], device=device),
        torch.tensor([-208.6974, -208.2225, -525.7090, -490.2226, -196.4263, -189.3733], device=device),
        torch.tensor([-208.6373, -208.0038, -526.3684, -489.6016, -196.4626, -189.1139], device=device),
        torch.tensor([-23.8717, -24.0988, -38.9287, -40.3180, -50.0000, -50.0000], device=device),
        torch.tensor([-23.4423, -24.2216, -38.7704, -40.1146, -50.0000, -50.0000], device=device),
        torch.tensor([-23.0976, -25.1244, -39.0613, -40.2598, -50.0000, -50.0000], device=device),
        torch.tensor([0.1839, 0.1867, 0.2077, 0.1884, 0.1000, 0.1000], device=device),
        torch.tensor([0.1855, 0.1848, 0.2006, 0.1825, 0.1000, 0.1000], device=device),
        torch.tensor([0.1871, 0.1838, 0.2046, 0.1884, 0.1000, 0.1000], device=device),
        torch.tensor([16.7290, 16.6504, 36.1984, 38.5769, 50.0000, 50.0000], device=device),
        torch.tensor([16.7207, 16.6423, 36.0976, 38.4506, 50.0000, 50.0000], device=device),
        torch.tensor([16.6505, 16.6921, 36.1323, 38.4492, 50.0000, 50.0000], device=device),
    ]

    variability = [
        torch.tensor([157.6598, 157.6590, 308.6288, 315.2618,  14.6688,  13.0800], device=device),
        torch.tensor([157.6627, 157.6609, 308.6041, 315.3110,  11.5919,  13.0425], device=device),
        torch.tensor([157.6628, 157.6606, 308.6055, 315.2387,  17.3468,  13.0231], device=device),
        torch.tensor([8.5719,  8.5872, 11.9652, 12.0870,  0.7690,  0.6359], device=device),
        torch.tensor([8.6150,  8.5830, 12.0298, 12.1981,  0.9564,  0.5670], device=device),
        torch.tensor([8.6406,  8.5171, 11.9606, 12.1388,  0.8187,  0.6091], device=device),
        torch.tensor([0.0105, 0.0161, 0.1467, 0.1491, 0.0266, 0.0143], device=device),
        torch.tensor([0.0147, 0.0132, 0.1355, 0.1383, 0.0113, 0.0211], device=device),
        torch.tensor([0.0171, 0.0105, 0.1417, 0.1491, 0.0258, 0.0166], device=device),
        torch.tensor([0.5982,  0.4684, 10.2633, 10.7480,  0.5587,  0.5640], device=device),
        torch.tensor([0.5880,  0.4684, 10.2950, 10.8114,  0.6105,  0.7380], device=device),
        torch.tensor([0.4684,  0.5433, 10.2653, 10.8260,  0.4684,  0.4684], device=device),
    ]

    """
    lbs = [i * 0.25 if i.min() >= 0 else i * 4.0 for i in real_initial_values]
    ubs = [i * 4.0 if i.min() >= 0 else i * 0.25 for i in real_initial_values]
    scales = scale_producer(names, lbs, ubs)
    """

    lbs, ubs, scales = None, None, None


    eps = torch.tensor(
        [
            2.9392e-06,
            2.9392e-06,
            8.5404e-05,
            2.9392e-06,
            2.9392e-06,
            2.9392e-06,
            2.9392e-06,
        ],
        device=device,
    )

    for etype in etypes[2:]:

        _ = predict(
            f,
            names,
            Tcontrol,
            posteriors,
            variability,
            scales,
            lbs,
            ubs,
            eps,
            model_maker,
            nsamples,
            confidence,
            etype,
            T_min=15.0 + 273.15,
            T_max=762.0 + 273.15,
            pscaling=False,
            acord_load=False,
        )

