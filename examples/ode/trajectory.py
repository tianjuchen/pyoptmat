#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import Parameter
import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoDelta, AutoNormal, init_to_mean,  AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, JitTrace_ELBO
import pyro.optim as optim
import pyro.distributions.constraints as constraints
from pyro import poutine

from tqdm import tqdm

import sys
sys.path.append('../..')
from pyoptmat import ode

g = torch.tensor(0.1)

v_loc_act = 2.0
v_scale_act = 0.03
a_loc_act = 0.7
a_scale_act = 0.04

v_loc_prior = 1.5
v_scale_prior = 0.1
a_loc_prior = 0.3
a_scale_prior = 0.1

eps_act = 0.05
eps_prior = 0.1 # Just measure variance in data...
jit_mode = False

def model_act(times):
  """
    times: ntime x nbatch
    trajectories: ntime x nbatch x 2
  """
  v = pyro.sample("v", dist.Normal(v_loc_act, v_scale_act))
  a = pyro.sample("a", dist.Normal(a_loc_act, a_scale_act))

  simulated = pyro.sample("data", dist.Normal(torch.stack((
    v * torch.cos(a) * times, 
    v * torch.sin(a) * times - 0.5 * g * times**2.0)).T, eps_act))

  return simulated

class Integrator(pyro.nn.PyroModule):
  def __init__(self, eqn, y0, extra_params = []):
    super().__init__()
    self.eqn = eqn
    self.y0 = y0
    self.extra_params = extra_params

  def forward(self, times):
    return ode.odeint_adjoint(self.eqn, self.y0, times, 
        jit_mode = jit_mode , extra_params = self.extra_params)

class ODE(pyro.nn.PyroModule):
  def __init__(self, v, a):
    super().__init__()
    self.v = v
    self.a = a

  def forward(self, t, y):
    f = torch.empty(y.shape)

    # Acceleration
    f[...,0] = self.v * torch.cos(self.a)
    f[...,1] = self.v * torch.sin(self.a) - g * t
    
    # Nice ODE lol
    df = torch.zeros(y.shape + y.shape[1:])

    return f, df

class Model(pyro.nn.PyroModule):
  def __init__(self, maker, names, loc_priors, scale_priors,
      loc_suffix = "_loc", scale_suffix = "_scale", param_suffix = "_param",
      bot_suffix = "_bot"):
    super().__init__()

    self.maker = maker

    self.loc_suffix = loc_suffix
    self.scale_suffix = scale_suffix
    self.param_suffix = param_suffix
    self.bot_suffix = bot_suffix
    self.loc_priors = loc_priors
    self.scale_priors = scale_priors
    self.eps_prior = eps_prior
    self.names = names

    # Setup both levels of distributions
    self.bot_vars = names
    self.top_vars = []
    for var, loc, scale in zip(names, loc_priors, scale_priors):
      setattr(self, var + loc_suffix, PyroSample(dist.Normal(loc, scale)))
      self.top_vars.append(var + loc_suffix)
      setattr(self, var + scale_suffix, PyroSample(dist.LogNormal(scale,1)))
      self.top_vars.append(var + scale_suffix)
      setattr(self, var, PyroSample(
        lambda self, var = var: dist.Normal(getattr(self, var + loc_suffix), 
          getattr(self, var + scale_suffix))))
    
    # Setup noise
    self.eps = PyroSample(dist.LogNormal(eps_prior,1))

    self.extra_param_names = []

  def forward(self, times, actual = None):
    y0 = torch.zeros((times.shape[1],) + (2,))
    
    curr = self.sample_top()
    eps = self.eps

    with pyro.plate("trials", times.shape[1]):
      bmodel = self.maker(*self.sample_bot(), 
          extra_params = self.gen_extra())
      simulated = bmodel(times)
      with pyro.plate("time", times.shape[0]):
        pyro.sample("obs", dist.Normal(simulated, eps).to_event(1), obs = actual)

    return simulated

  def sample_top(self):
    return [getattr(self, name) for name in self.top_vars]

  def sample_bot(self):
    return [getattr(self, name) for name in self.bot_vars]

  def make_guide(self):
    def guide(times, actual = None):
      top_loc_samples = []
      top_scale_samples = []

      for name, loc, scale in zip(self.names, self.loc_priors, self.scale_priors):
        loc_param = pyro.param(name + self.loc_suffix + self.param_suffix,
            torch.tensor(loc))
        scale_param = pyro.param(name + self.scale_suffix + self.param_suffix,
            torch.tensor(scale), constraint = constraints.positive)

        top_loc_samples.append(pyro.sample(name + self.loc_suffix, 
          dist.Delta(loc_param)))
        top_scale_samples.append(pyro.sample(name + self.scale_suffix, 
          dist.Delta(scale_param)))

      eps_param = pyro.param("eps" + self.param_suffix, 
          torch.tensor(self.eps_prior), constraint = constraints.positive)
      eps_sample = pyro.sample("eps", dist.Delta(eps_param))
      
      with pyro.plate("trials", times.shape[1]):
        for name, loc_sample, scale_sample, loc_prior, scale_prior in zip(self.names, top_loc_samples,
            top_scale_samples, self.loc_priors, self.scale_priors):
          ll_param = pyro.param(name + self.param_suffix, 
              torch.ones(times.shape[1]) * torch.tensor(loc_prior))
          pyro.sample(name, dist.Delta(ll_param))

    self.extra_param_names = [var + self.param_suffix for var in self.names]

    return guide

  def gen_extra(self):
    return [pyro.param(name).unconstrained() for name in self.extra_param_names]

if __name__ == "__main__":
  nsamples = 50

  tmax = 20.0
  tnum = 100

  time = torch.linspace(0, tmax, tnum)
  times = torch.empty(tnum, nsamples)
  data = torch.empty(tnum, nsamples, 2)
  
  with torch.no_grad():
    for i in range(nsamples):
      times[:,i] = time
      data[:,i] = model_act(time)

  plt.plot(data[:,:,0], data[:,:,1])
  plt.show()
  
  # MAP
  print("MAP")
  pyro.clear_param_store()

  def maker(v, a, **kwargs):
    return Integrator(ODE(v,a), torch.zeros(nsamples, 2), **kwargs)

  # Setup the model
  model = Model(maker, ["v", "a"], [v_loc_prior, a_loc_prior], [v_scale_prior, a_scale_prior])

  lr = 5.0e-3
  niter = 500
  num_samples = 1

  guide = model.make_guide()
  
  # Init guide
  guide(times)

  optimizer = optim.ClippedAdam({"lr": lr})
  if jit_mode:
    l = JitTrace_ELBO(num_particles = num_samples)
  else:
    l = Trace_ELBO(num_particles = num_samples)

  svi = SVI(model, guide, optimizer, loss = l)
  
  t = tqdm(range(niter))
  loss_hist = []
  for i in t:
    loss = svi.step(times, data)
    loss_hist.append(loss)
    t.set_description("Loss: %3.2e" % loss)

  print("Inferred distributions:")
  print("Velocity mean: %4.3f, actual %4.3f" % (pyro.param("v_loc_param").data, v_loc_act))
  print("Velocity scale: %4.3f, actual %4.3f" % (pyro.param("v_scale_param").data, v_scale_act))
  print("Angle mean: %4.3f, actual %4.3f" % (pyro.param("a_loc_param").data, a_loc_act))
  print("Angle scale: %4.3f, actual %4.3f" % (pyro.param("a_scale_param").data, a_scale_act))
  print("White noise: %4.3f, actual %4.3f" % (pyro.param("eps_param").data, eps_act))

  plt.plot(loss_hist)
  plt.show()

  print("")

  nsample = 1
  predict = Predictive(model, guide = guide, num_samples = nsample,
      return_sites=("obs",))
  with torch.no_grad():
    samples = predict(times)["obs"]
    min_x, _ = torch.min(samples[0,:,:,0], 1)
    max_x, _ = torch.max(samples[0,:,:,0], 1)
    min_y, _ = torch.min(samples[0,:,:,1], 1)
    max_y, _ = torch.max(samples[0,:,:,1], 1)

  plt.plot(times, data[:,:,1], 'k-', lw = 0.5)
  plt.fill_between(time, min_y, max_y, alpha = 0.75)
  plt.show()

  plt.plot(times, data[:,:,0], 'k-', lw = 0.5)
  plt.fill_between(time, min_x, max_x, alpha = 0.75)
  plt.show()
