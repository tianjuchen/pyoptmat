#!/usr/bin/env python3

import os, sys
sys.path.append('../../../..')
sys.path.append('..')

import numpy as np
import numpy.random as ra
import xarray as xr
import torch
import torch.distributions.constraints as constraints
from maker import make_model, load_data, sf
from pyoptmat import optimize
from tqdm import tqdm
import pyro
from pyro.infer import Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO, SVI, config_enumerate
from pyro.infer.mcmc import MCMC, NUTS
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
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, device = device,
      use_adjoint = True, jit_mode = jit_mode, 
      **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.15
  nsamples = 10 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)

  # this is for running the notebook in our testing framework
  #n_steps = 12000
  #pyro.set_rng_seed(2)

  # clear the param store in case we're in a REPL
  pyro.clear_param_store()
  
  mean_n, mean_eta, mean_s0, mean_R, mean_d = (torch.tensor(ra.random()), torch.tensor(ra.random()),
                    torch.tensor(ra.random()), torch.tensor(ra.random()), torch.tensor(ra.random()))
  scale_n, scale_eta, scale_s0, scale_R, scale_d = (torch.tensor(ra.random()), torch.tensor(ra.random()),
                    torch.tensor(ra.random()), torch.tensor(ra.random()), torch.tensor(ra.random()))
  top_scale = torch.tensor(0.15)
  eps_prior = 1.0e-4
  mean_prior = torch.tensor(0.1)
  
  def model(times, strains, temperatures, true_stresses):
    n_loc = pyro.sample("n_loc", dist.Normal(mean_n,scale_n).to_event(mean_n.dim()))
    eta_loc = pyro.sample("eta_loc", dist.Normal(mean_eta,scale_eta).to_event(mean_eta.dim()))
    s0_loc = pyro.sample("s0_loc", dist.Normal(mean_s0,scale_s0).to_event(mean_s0.dim()))
    R_loc = pyro.sample("R_loc", dist.Normal(mean_R,scale_R).to_event(mean_R.dim()))
    d_loc = pyro.sample("d_loc", dist.Normal(mean_d,scale_d).to_event(mean_d.dim()))
    n_scale = pyro.sample("n_scale", dist.HalfNormal(top_scale).to_event(top_scale.dim()))
    eta_scale = pyro.sample("eta_scale", dist.HalfNormal(top_scale).to_event(top_scale.dim()))
    s0_scale = pyro.sample("s0_scale", dist.HalfNormal(top_scale).to_event(top_scale.dim()))
    R_scale = pyro.sample("R_scale", dist.HalfNormal(top_scale).to_event(top_scale.dim()))
    d_scale = pyro.sample("d_scale", dist.HalfNormal(top_scale).to_event(top_scale.dim()))
    eps = pyro.sample("eps", dist.HalfNormal(torch.tensor(eps_prior)))
   
    with pyro.plate("trials", times.shape[1]):
      n = pyro.sample("n", dist.Normal(n_loc, n_scale).to_event(mean_n.dim()))
      eta = pyro.sample("eta", dist.Normal(eta_loc, eta_scale).to_event(mean_eta.dim()))
      s0 = pyro.sample("s0", dist.Normal(s0_loc, s0_scale).to_event(mean_s0.dim()))
      R = pyro.sample("R", dist.Normal(R_loc, R_scale).to_event(mean_R.dim()))
      d = pyro.sample("d", dist.Normal(d_loc, d_scale).to_event(mean_d.dim()))
      bmodel = make(n, eta, s0, R, d)
      sim_res = bmodel.solve_strain(times, strains, temperatures)[:,:,0]
      with pyro.plate("time", times.shape[0]):
        pyro.sample("obs", dist.Normal(sim_res, eps), obs = true_stresses)      
        
  def guide(times, strains, temperatures, true_stresses):
    n_top_loc = pyro.param("n_top_loc", mean_prior)
    eta_top_loc = pyro.param("eta_top_loc", mean_prior)
    s0_top_loc = pyro.param("s0_top_loc", mean_prior)
    R_top_loc = pyro.param("R_top_loc", mean_prior)
    d_top_loc = pyro.param("d_top_loc", mean_prior)
    n_top_scale = pyro.param("n_top_scale", top_scale, 
        constraint = constraints.positive)
    eta_top_scale = pyro.param("eta_top_scale", top_scale, 
        constraint = constraints.positive)
    s0_top_scale = pyro.param("s0_top_scale", top_scale, 
        constraint = constraints.positive)
    R_top_scale = pyro.param("R_top_scale", top_scale, 
        constraint = constraints.positive)
    d_top_scale = pyro.param("d_top_scale", top_scale, 
        constraint = constraints.positive)
    
    eps_param = pyro.param("eps_param", torch.tensor(eps_prior), 
                constraint = constraints.positive)
    
    n_loc = pyro.sample("n_loc", dist.Delta(n_top_loc,).to_event(mean_n.dim()))
    eta_loc = pyro.sample("eta_loc", dist.Delta(eta_top_loc).to_event(mean_eta.dim()))
    s0_loc = pyro.sample("s0_loc", dist.Delta(s0_top_loc).to_event(mean_s0.dim()))
    R_loc = pyro.sample("R_loc", dist.Delta(R_top_loc).to_event(mean_R.dim()))
    d_loc = pyro.sample("d_loc", dist.Delta(d_top_loc).to_event(mean_d.dim()))
    n_scale = pyro.sample("n_scale", dist.Delta(n_top_scale).to_event(top_scale.dim()))
    eta_scale = pyro.sample("eta_scale", dist.Delta(eta_top_scale).to_event(top_scale.dim()))
    s0_scale = pyro.sample("s0_scale", dist.Delta(s0_top_scale).to_event(top_scale.dim()))
    R_scale = pyro.sample("R_scale", dist.Delta(R_top_scale).to_event(top_scale.dim()))
    d_scale = pyro.sample("d_scale", dist.Delta(d_top_scale).to_event(top_scale.dim()))
    eps = pyro.sample("eps", dist.Delta(torch.tensor(eps_param)))
       
    with pyro.plate("trials", times.shape[1]):
      n_bot_loc = pyro.param("n_bot_loc", mean_prior)
      eta_bot_loc = pyro.param("eta_bot_loc", mean_prior)
      s0_bot_loc = pyro.param("s0_bot_loc", mean_prior)
      R_bot_loc = pyro.param("R_bot_loc", mean_prior)
      d_bot_loc = pyro.param("d_bot_loc", mean_prior)    
      n = pyro.sample("n", dist.Delta(n_bot_loc).to_event(mean_n.dim()))
      eta = pyro.sample("eta", dist.Delta(eta_bot_loc).to_event(mean_eta.dim()))
      s0 = pyro.sample("s0", dist.Delta(s0_bot_loc).to_event(mean_s0.dim()))
      R = pyro.sample("R", dist.Delta(R_bot_loc).to_event(mean_R.dim()))
      d = pyro.sample("d", dist.Delta(d_bot_loc).to_event(mean_d.dim()))

  use_svi = True
  use_mcmc = False

  if use_svi:
    niter = 500
    elbo = Trace_ELBO()
    # elbo = JitTrace_ELBO()
    svi = SVI(model, config_enumerate(guide), Adam({'lr': 0.01}), elbo)
    t = tqdm(range(niter), total = niter, desc = "Loss:    ")
    loss_hist = []
    for i in t:
      loss = svi.step(times, strains, temperatures, true_stresses)
      loss_hist.append(loss)
      t.set_description("Loss %3.2e" % loss)

    print("Pyro Inferred Parameters are:")
    for name, value in pyro.get_param_store().items():
      print(name, pyro.param(name))
    print("")
  elif use_mcmc:
  
    nuts_kernel = NUTS(model)
    pyro.set_rng_seed(1)
    mcmc_run = MCMC(nuts_kernel, num_samples=100).run(times, strains, temperatures, true_stresses)
    
    #nuts_kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True)
    #pyro.set_rng_seed(1)
    #mcmc_run = MCMC(nuts_kernel, num_samples=500).run(times, strains, temperatures, true_stresses)
