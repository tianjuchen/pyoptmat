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
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, device = device,
      use_adjoint = False, jit_mode = jit_mode, 
      **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.15
  nsamples = 20 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)
  print(times.size())
  # this is for running the notebook in our testing framework
  n_steps = 1000
  pyro.set_rng_seed(2)

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
 
  """
  def guide(times, strains, temperatures, true_stresses, index):
    n_top_loc = pyro.param("n_top_loc_{}".format(index), mean_prior)
    eta_top_loc = pyro.param("eta_top_loc_{}".format(index), mean_prior)
    s0_top_loc = pyro.param("s0_top_loc_{}".format(index), mean_prior)
    R_top_loc = pyro.param("R_top_loc_{}".format(index), mean_prior)
    d_top_loc = pyro.param("d_top_loc_{}".format(index), mean_prior)
    n_top_scale = pyro.param("n_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    eta_top_scale = pyro.param("eta_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    s0_top_scale = pyro.param("s0_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    R_top_scale = pyro.param("R_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    d_top_scale = pyro.param("d_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    
    eps_param = pyro.param("eps_param_{}".format(index), torch.tensor(eps_prior), 
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
      n_bot_loc = pyro.param("n_bot_loc_{}".format(index), mean_prior)
      eta_bot_loc = pyro.param("eta_bot_loc_{}".format(index), mean_prior)
      s0_bot_loc = pyro.param("s0_bot_loc_{}".format(index), mean_prior)
      R_bot_loc = pyro.param("R_bot_loc_{}".format(index), mean_prior)
      d_bot_loc = pyro.param("d_bot_loc_{}".format(index), mean_prior)  
      n = pyro.sample("n", dist.Delta(n_bot_loc).to_event(mean_n.dim()))
      eta = pyro.sample("eta", dist.Delta(eta_bot_loc).to_event(mean_eta.dim()))
      s0 = pyro.sample("s0", dist.Delta(s0_bot_loc).to_event(mean_s0.dim()))
      R = pyro.sample("R", dist.Delta(R_bot_loc).to_event(mean_R.dim()))
      d = pyro.sample("d", dist.Delta(d_bot_loc).to_event(mean_d.dim()))
  """
  def guide(times, strains, temperatures, true_stresses, index):
    n_top_loc = pyro.param("n_top_loc_{}".format(index), mean_prior)
    eta_top_loc = pyro.param("eta_top_loc_{}".format(index), mean_prior)
    s0_top_loc = pyro.param("s0_top_loc_{}".format(index), mean_prior)
    R_top_loc = pyro.param("R_top_loc_{}".format(index), mean_prior)
    d_top_loc = pyro.param("d_top_loc_{}".format(index), mean_prior)
    n_top_scale = pyro.param("n_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    eta_top_scale = pyro.param("eta_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    s0_top_scale = pyro.param("s0_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    R_top_scale = pyro.param("R_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    d_top_scale = pyro.param("d_top_scale_{}".format(index), top_scale, 
        constraint = constraints.positive)
    
    eps_param = pyro.param("eps_{}".format(index), torch.tensor(eps_prior), 
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
      n_bot_loc = pyro.param("n_bot_loc_{}".format(index), mean_prior)
      eta_bot_loc = pyro.param("eta_bot_loc_{}".format(index), mean_prior)
      s0_bot_loc = pyro.param("s0_bot_loc_{}".format(index), mean_prior)
      R_bot_loc = pyro.param("R_bot_loc_{}".format(index), mean_prior)
      d_bot_loc = pyro.param("d_bot_loc_{}".format(index), mean_prior) 
      
      n_bot_scale = pyro.param("n_bot_scale_{}".format(index), top_scale,
                    constraint = constraints.positive)
      eta_bot_scale = pyro.param("eta_bot_scale_{}".format(index), top_scale,
                    constraint = constraints.positive)
      s0_bot_scale = pyro.param("s0_bot_scale_{}".format(index), top_scale,
                    constraint = constraints.positive)
      R_bot_scale = pyro.param("R_bot_scale_{}".format(index), top_scale,
                    constraint = constraints.positive)
      d_bot_scale = pyro.param("d_bot_scale_{}".format(index), top_scale,
                    constraint = constraints.positive) 
      n = pyro.sample("n", dist.Normal(n_bot_loc, n_bot_scale).to_event(mean_n.dim()))
      eta = pyro.sample("eta", dist.Normal(eta_bot_loc, eta_bot_scale).to_event(mean_eta.dim()))
      s0 = pyro.sample("s0", dist.Normal(s0_bot_loc, s0_bot_scale).to_event(mean_s0.dim()))
      R = pyro.sample("R", dist.Normal(R_bot_loc, R_bot_scale).to_event(mean_R.dim()))
      d = pyro.sample("d", dist.Normal(d_bot_loc, d_bot_scale).to_event(mean_d.dim()))


  def relbo(model, guide, *args, **kwargs):
    approximation = kwargs.pop('approximation')

    # We first compute the elbo, but record a guide trace for use below.
    traced_guide = trace(guide)
    elbo = pyro.infer.Trace_ELBO()
    loss_fn = elbo.differentiable_loss(model, traced_guide, *args, **kwargs)

    # We do not want to update parameters of previously fitted components
    # and thus block all parameters in the approximation apart from z.
    guide_trace = traced_guide.trace
    replayed_approximation = trace(replay(block(approximation, expose=["n","eta","s0","R","d"]), 
                             guide_trace))
    approximation_trace = replayed_approximation.get_trace(*args, **kwargs)

    relbo = -loss_fn - approximation_trace.log_prob_sum()

    # By convention, the negative (R)ELBO is returned.
    return -relbo
        
  def approximation(times, strains, temperatures, true_stresses, components, weights):
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    result = components[assignment](times, strains, temperatures, true_stresses)
    return result        
        
  def boosting_bbvi():
    # T=2
    n_iterations = 2
    initial_approximation = partial(guide, index=0)
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = partial(approximation, components=components, weights=weights)

    locs = [0]
    scales = [0]

    for t in range(1, n_iterations + 1):

        # Create guide that only takes data as argument
        wrapped_guide = partial(guide, index=t)
        losses = []

        adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        # Pass our custom RELBO to SVI as the loss function.
        svi = SVI(model, wrapped_guide, optimizer, loss=relbo)
        title = tqdm(range(n_steps), total = n_steps, desc = "Loss:    ")
        for step in title:
            # Pass the existing approximation to SVI.
            loss = svi.step(times, strains, temperatures, true_stresses, approximation=wrapped_approximation)
            losses.append(loss)
            title.set_description("Loss %3.2e" % loss)
            if step % 100 == 0:
                print('.', end=' ')

        # Update the list of approximation components.
        components.append(wrapped_guide)

        # Set new mixture weight.
        new_weight = 2 / (t + 1)

        # In this specific case, we set the mixture weight of the second component to 0.5.
        if t == 2:
            new_weight = 0.5
        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        # Update the approximation
        wrapped_approximation = partial(approximation, components=components, weights=weights)



        print("Pyro Inferred Parameters:")
        for name, value in pyro.get_param_store().items():
          print(name, pyro.param(name))
        print("")
        """
        print('Parameters of component {}:'.format(t))
        scale = pyro.param("scale_{}".format(t)).item()
        scales.append(scale)
        loc = pyro.param("loc_{}".format(t)).item()
        locs.append(loc)
        print('loc = {}'.format(loc))
        print('scale = {}'.format(scale))
        """
    """
    # Plot the resulting approximation
    X = np.arange(-10, 10, 0.1)
    pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    total_approximation = np.zeros(X.shape)
    for i in range(1, n_iterations + 1):
        Y = weights[i].item() * scipy.stats.norm.pdf((X - locs[i]) / scales[i])
        pyplot.plot(X, Y)
        total_approximation += Y
    pyplot.plot(X, total_approximation)
    pyplot.plot(data.data.numpy(), np.zeros(len(data)), 'k*')
    pyplot.title('Approximation of posterior over z')
    pyplot.ylabel('probability density')
    pyplot.show()
    """

  run_bbvi = boosting_bbvi()      
