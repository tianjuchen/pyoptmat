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

import matplotlib.pyplot as plt

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

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, use_adjoint = True,
      device = device, **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  sample_size = [5, 10, 20, 30]
  scale = 0.15
  nsamples = 30 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)
  

  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d"]
  ics = [0.46, 0.46, 0.53, 0.44, 0.45]

  initials = []
  for i in range(len(names)):
    initials.append(torch.tensor(ics[i]))
  
  print(initials)
  
  # 9) evaluate the performance of the optimziation
  with torch.no_grad():
    eva_stress = make(*initials).solve_strain(times, strains, temperatures)[:,:,0]
  
  print(eva_stress.size())
    
  # ) 10 evaluate the deterministic performance
  ntrue = true_stresses.shape[1]
  max_true, _ = true_stresses.kthvalue(int(ntrue), dim=1)
  min_true, _ = true_stresses.kthvalue(int(1), dim=1)
  mean_true = torch.mean(true_stresses, 1)
  mean_optimize = torch.mean(eva_stress, 1)
  plt.figure()
  plt.plot(strains[:,0].numpy(), mean_true.numpy(), 'k-', lw=2.0,
      label = "Actual mean")
  plt.fill_between(strains[:,0].numpy(), min_true.numpy(), max_true.numpy(), color = 'k',
      alpha = 0.4, label = "Actual range")
  plt.plot(strains[:,0].numpy(), mean_optimize.numpy(), 'b--', lw=2.0, label = 'optimize')
  plt.xlabel("Strain (mm/mm)", fontsize=16)
  plt.ylabel("Stress (MPa)", fontsize=16)
  # plt.xticks(fontsize=16)
  # plt.yticks(fontsize=16)
  plt.tight_layout()
  plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
  plt.grid(True)
  plt.savefig("Deterministic-newton-size-{}.png".format(nsamples))
  plt.show()
  plt.close()  
  