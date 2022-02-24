#!/usr/bin/env python3

import os, sys
sys.path.append('../../../..')
sys.path.append('..')

import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import torch.distributions.constraints as constraints
from maker import make_model, load_data, sf, unscale_model
from pyoptmat import optimize
from tqdm import tqdm
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.easyguide import easy_guide
import pyro.optim as optim
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.stats
from pyro.poutine import block, replay, trace
from pyro.optim import Adam
import pyro.distributions as dist

import warnings
warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# Run on CPU (home machine GPU is eh)
dev = "cpu"
device = torch.device(dev)

jit_mode = False

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
  return unscale_model(torch.tensor(150000.0), n, eta, s0, R, d, device = device,
      use_adjoint = False, jit_mode = jit_mode, 
      **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.01
  nsamples = 10 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)
  # this is for running the notebook in our testing framework
  pyro.set_rng_seed(2)

  # clear the param store in case we're in a REPL
  pyro.clear_param_store()
  """
  mean_n, mean_eta, mean_s0, mean_R, mean_d = (torch.tensor(ra.random()), torch.tensor(ra.random()),
                    torch.tensor(ra.random()), torch.tensor(ra.random()), torch.tensor(ra.random()))
  scale_n, scale_eta, scale_s0, scale_R, scale_d = (torch.tensor(ra.random()), torch.tensor(ra.random()),
                    torch.tensor(ra.random()), torch.tensor(ra.random()), torch.tensor(ra.random()))
  """
  top_scale = torch.tensor(5.0)
  eps_prior = 1.0e-4
  mean_n, mean_eta, mean_s0, mean_R, mean_d = map(torch.tensor, (10.0, 200.0, 30.0, 600.0, 50.0))

  def model(times, strains, temperatures, true_stresses):
    n_loc = pyro.param("n_loc", mean_n, constraints.interval(1.0, 20.0))
    eta_loc = pyro.param("eta_loc", mean_eta, constraints.interval(100.0, 500.0))
    s0_loc = pyro.param("s0_loc", mean_s0, constraints.interval(10.0, 100.0))
    R_loc = pyro.param("R_loc", mean_R, constraints.interval(100.0, 1000.0))
    d_loc = pyro.param("d_loc", mean_d, constraints.interval(1.0, 100.0))
    n_scale = pyro.param("n_scale", top_scale, constraint = constraints.positive)
    eta_scale = pyro.param("eta_scale", top_scale, constraint = constraints.positive)
    s0_scale = pyro.param("s0_scale", top_scale, constraint = constraints.positive)
    R_scale = pyro.param("R_scale", top_scale, constraint = constraints.positive)
    d_scale = pyro.param("d_scale", top_scale, constraint = constraints.positive)
    eps = pyro.param("eps", torch.tensor(eps_prior), constraint = constraints.positive)
   
    with pyro.plate("trials", times.shape[1]):
      n = pyro.param("n", mean_n, constraints.interval(1.0, 20.0))
      eta = pyro.param("eta", mean_eta, constraints.interval(100.0, 500.0))
      s0 = pyro.param("s0", mean_s0, constraints.interval(10.0, 100.0))
      R = pyro.param("R", mean_R, constraints.interval(100.0, 1000.0))
      d = pyro.param("d", mean_d, constraints.interval(1.0, 100.0))
      bmodel = make(n, eta, s0, R, d)
      sim_res = bmodel.solve_strain(times, strains, temperatures)[:,:,0]
      with pyro.plate("time", times.shape[0]):
        pyro.sample("obs", dist.Normal(sim_res, eps), obs = true_stresses)      


  def guide(times, strains, temperatures, true_stresses):
    pass

  def train(model, guide, lr=0.01, n_steps=200):
    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for step in range(n_steps):
      loss = svi.step(times, strains, temperatures, true_stresses)
      if step % 20 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))
        
    print("Pyro Inferred Parameters:")
    for name, value in pyro.get_param_store().items():
      print(name, pyro.param(name))
    print("")

  train(model, guide)