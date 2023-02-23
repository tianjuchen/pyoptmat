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
from pyoptmat import models, flowrules, hardening, optimize
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
device = torch.device(dev)

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0


def Arrhenius(p, T, Q = torch.tensor(-100.0)):
    
    return p * torch.exp(-Q/T)

if __name__ == "__main__":

    Ts = torch.tensor([25.0, 300.0, 500.0, 1000000.0]) + 273.15
    Es = Arrhenius(E_true, Ts)
    Rs = Arrhenius(R_true, Ts)
    ds = Arrhenius(d_true, Ts)
    ns = Arrhenius(n_true, Ts)
    etas = Arrhenius(eta_true, Ts)
    s0s = Arrhenius(s0_true, Ts)
    
    print(Es)
    print(Rs)
    print(ds)
    print(ns)
    print(etas)
    print(s0s)
    
    


