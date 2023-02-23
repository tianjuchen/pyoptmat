#!/usr/bin/env python3

"""
    Helper functions for the structural material model inference with
    tension tests examples.
"""

import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
from pyoptmat import models, flowrules, hardening, optimize, scaling, temperature
from pyoptmat.temperature import ConstantParameter as CP

from tqdm import tqdm

import tqdm

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0

# Scale factor used in the model definition
sf = 0.5

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


scalefns = [
    scaling.BoundedScalingFunction(
            torch.tensor(E_true * (1 - sf), device=device), 
            torch.tensor(E_true * (1 + sf), device=device)
        ),
    scaling.BoundedScalingFunction(
            torch.tensor(n_true * (1 - sf), device=device), 
            torch.tensor(n_true * (1 + sf), device=device)
        ),
    scaling.BoundedScalingFunction(
            torch.tensor(eta_true * (1 - sf), device=device), 
            torch.tensor(eta_true * (1 + sf), device=device)
        ),
    scaling.BoundedScalingFunction(
            torch.tensor(s0_true * (1 - sf), device=device), 
            torch.tensor(s0_true * (1 + sf), device=device)
        ),
    scaling.BoundedScalingFunction(
            torch.tensor(R_true * (1 - sf), device=device), 
            torch.tensor(R_true * (1 + sf), device=device)
        ),
    scaling.BoundedScalingFunction(
            torch.tensor(d_true * (1 - sf), device=device), 
            torch.tensor(d_true * (1 + sf), device=device)
        ),
]


def make_model(E, n, eta, s0, R, d, scale_functions=None, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """
    Q = torch.tensor(-100.0)

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

    Et = temperature.ArrheniusScaling(E, Q, A_scale=E_scale)
    nt= temperature.ArrheniusScaling(n, Q, A_scale=n_scale)
    etat = temperature.ArrheniusScaling(eta, Q, A_scale=eta_scale)
    s0t = temperature.ArrheniusScaling(s0, Q, A_scale=s0_scale)
    Rt = temperature.ArrheniusScaling(R, Q, A_scale=R_scale)
    dt = temperature.ArrheniusScaling(d, Q, A_scale=d_scale)
    
    
    isotropic = hardening.VoceIsotropicHardeningModel(Rt, dt)
    kinematic = hardening.NoKinematicHardeningModel()
    flowrule = flowrules.IsoKinViscoplasticity(
        nt, etat, s0t,
        isotropic,
        kinematic,
    )
    model = models.InelasticModel(Et, flowrule)

    return models.ModelIntegrator(model, **kwargs)



def grid_model(E, n, eta, s0, R, d, scale_functions=None, Tcontrol=None, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """

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

    Et = temperature.PiecewiseScaling(Tcontrol, E, values_scale_fn=E_scale)
    nt= temperature.PiecewiseScaling(Tcontrol, n, values_scale_fn=n_scale)
    etat = temperature.PiecewiseScaling(Tcontrol, eta, values_scale_fn=eta_scale)
    s0t = temperature.PiecewiseScaling(Tcontrol, s0, values_scale_fn=s0_scale)
    Rt = temperature.PiecewiseScaling(Tcontrol, R, values_scale_fn=R_scale)
    dt = temperature.PiecewiseScaling(Tcontrol, d, values_scale_fn=d_scale)
    
    
    isotropic = hardening.VoceIsotropicHardeningModel(Rt, dt)
    kinematic = hardening.NoKinematicHardeningModel()
    flowrule = flowrules.IsoKinViscoplasticity(
        nt, etat, s0t,
        isotropic,
        kinematic,
    )
    model = models.InelasticModel(Et, flowrule)

    return models.ModelIntegrator(model, **kwargs)




def generate_input(erates, emax, ntime):
    """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
    """
    strain = torch.repeat_interleave(
        torch.linspace(0, emax, ntime, device=device)[None, :], len(erates), 0
    ).T.to(device)
    time = strain / erates

    return time, strain


def downsample(rawdata, nkeep, nrates, nsamples):
    """
    Return fewer than the whole number of samples for each strain rate
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )


def adding_noise(x, noise=100.0):
    return x + (noise**0.5)*torch.randn(x.shape)

if __name__ == "__main__":

    # Running this script will regenerate the data
    ntime = 200
    emax = 0.5
    erates = torch.logspace(-2, -4, 3, device=device)
    nrates = len(erates)
    nsamples = 50

    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    times, strains = generate_input(erates, emax, ntime)
    
    # tcontrol = torch.tensor([25.0, 300.0, 500.0]) + 273.15
    
    tcontrol = torch.randint(25, 500, (10,)) + 273.15
    print(tcontrol)
    # sys.exit("stop")
    
    full_temperatures = torch.empty((ntime, nrates, nsamples), device=device)
    for i in range(nsamples):
        indice = torch.tensor(ra.randint(len(tcontrol)))
        full_temperatures[:, :, i] = torch.ones_like(full_temperatures[:, :, i]) * tcontrol[indice]
    
    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, nrates, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)

        for i in tqdm.tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                scale_functions = scalefns,
            )

            with torch.no_grad():
                full_stresses[:, :, i] = adding_noise(model.solve_strain(
                    times, strains, full_temperatures[:, :, i]
                )[:, :, 0], noise=0.1)

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["tensile"] * (nsamples * len(erates)))
        controls = np.array(["strain"] * (nsamples * len(erates)))

        ds = xr.Dataset(
            {
                "time": (["ntime", "nexp"], full_times.flatten(-2, -1).cpu().numpy()),
                "strain": (
                    ["ntime", "nexp"],
                    full_strains.flatten(-2, -1).cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "nexp"],
                    full_stresses.flatten(-2, -1).cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "nexp"],
                    full_temperatures.cpu().flatten(-2, -1).numpy(),
                ),
                "cycle": (["ntime", "nexp"], full_cycles.flatten(-2, -1).cpu().numpy()),
                "type": (["nexp"], types),
                "control": (["nexp"], controls),
            },
            attrs={"scale": scale, "nrates": nrates, "nsamples": nsamples},
        )

        ds.to_netcdf("scale-%3.2f.nc" % scale)
