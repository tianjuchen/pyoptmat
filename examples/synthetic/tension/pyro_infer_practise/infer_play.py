#!/usr/bin/env python3

import sys
sys.path.append('../../../..')
sys.path.append('..')

import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize
from tqdm import tqdm

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.easyguide import easy_guide
import pyro.optim as optim
from pyro.infer import Predictive
import matplotlib.pyplot as plt
import seaborn as sns
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

  # 2) Setup names for each parameter and the priors
  names = ["n", "eta", "s0", "R", "d"]
  loc_loc_priors = [torch.tensor(ra.random(), device = device) for i in range(len(names))]
  loc_scale_priors = [torch.tensor(0.15, device = device) for i in range(len(names))]
  scale_scale_priors = [torch.tensor(0.15, device = device) for i in range(len(names))]

  eps = torch.tensor(1.0e-4, device = device)

  print("Initial parameter values:")
  print("\tloc loc\t\tloc scale\tscale scale")
  for n, llp, lsp, sp in zip(names, loc_loc_priors, loc_scale_priors, 
      scale_scale_priors):
    print("%s:\t%3.2f\t\t%3.2f\t\t%3.2f" % (n, llp, lsp, sp))
  print("")

  # 3) Create the actual model
  model = optimize.HierarchicalStatisticalModel(make, names, loc_loc_priors,
      loc_scale_priors, scale_scale_priors, eps).to(device)

  # 4) Get the guide
  guide = model.make_guide()
  
  # 5) Setup the optimizer and loss
  lr = 1.0e-2
  g = 1.0
  niter = 100
  lrd = g**(1.0 / niter)
  num_samples = 1
  
  optimizer = optim.ClippedAdam({"lr": lr, 'lrd': lrd})
  if jit_mode:
    ls = pyro.infer.JitTrace_ELBO(num_particles = num_samples)
  else:
    ls = pyro.infer.Trace_ELBO(num_particles = num_samples)

  svi = SVI(model, guide, optimizer, loss = ls)

  # 6) Infer!
  t = tqdm(range(niter), total = niter, desc = "Loss:    ")
  loss_hist = []
  for i in t:
    loss = svi.step(times, strains, temperatures, true_stresses)
    loss_hist.append(loss)
    t.set_description("Loss %3.2e" % loss)
 
  # 7) Print out results
  print("Pyro Inferred Parameters are:")
  for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
  print("")

  """
  # 8) Save some info
  np.savetxt("loss-history.txt", loss_hist)

  plt.figure()
  plt.loglog(loss_hist)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig("convergence.pdf")
  """

  # 9) plot the results
  #sites = ["n", "eta", "s0", "R", "d"]
  sites = ["n_loc", "eta_loc", "s0_loc", "R_loc", "d_loc"]
  num_samples = 100
  predictive_svi = Predictive(model, guide=guide, num_samples=num_samples)(times, strains, temperatures, None)
  for k, v in predictive_svi.items():
    print(f"{k}: {tuple(v.shape)}")
    
  fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20.0, 4.0)) # width , height
  fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
  print(len(axs.reshape(-1)))
  for i, ax in enumerate(axs.reshape(-1)):
    site = sites[i]
    sns.distplot(predictive_svi[site].detach().numpy(), ax=ax, label="SVI (Multivariate Normal)")
    ax.set_title(site)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right')
  plt.show()
  plt.close()

  # 10) Compare the cross-section of the Posterior Distribution
  fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 25))
  fig.suptitle("Cross-sections of the Posterior Distribution", fontsize=16)
   
  sns.kdeplot(predictive_svi["n"].detach().numpy(), predictive_svi["eta"].detach().numpy(), ax=axs[0], label="SVI (Multivariate Normal)")
  axs[0].set(xlabel="n", ylabel="eta")#, xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
  
  sns.kdeplot(predictive_svi["n"].detach().numpy(), predictive_svi["s0"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[1].set(xlabel="n", ylabel="s0")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
  
  sns.kdeplot(predictive_svi["n"].detach().numpy(), predictive_svi["R"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[2].set(xlabel="n", ylabel="R")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
 
  sns.kdeplot(predictive_svi["n"].detach().numpy(), predictive_svi["d"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[3].set(xlabel="n", ylabel="d")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["eta"].detach().numpy(), predictive_svi["s0"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[4].set(xlabel="eta", ylabel="s0")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["eta"].detach().numpy(), predictive_svi["R"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[5].set(xlabel="eta", ylabel="R")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["eta"].detach().numpy(), predictive_svi["d"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[6].set(xlabel="eta", ylabel="d")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["s0"].detach().numpy(), predictive_svi["R"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[7].set(xlabel="s0", ylabel="R")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["s0"].detach().numpy(), predictive_svi["d"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[8].set(xlabel="s0", ylabel="d")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  sns.kdeplot(predictive_svi["R"].detach().numpy(), predictive_svi["d"].detach().numpy(), ax=axs[1], label="SVI (Multivariate Normal)")
  axs[9].set(xlabel="R", ylabel="d")#, xlim=(-0.45, 0.05), ylim=(-0.15, 0.8)) 
  
  handles, labels = axs[1].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right');
  plt.show()
  plt.close()

