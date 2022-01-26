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
    # Load the data for the variance of interest,
    scale = 0.15
    samples = [5, 30] # at each strain rate
    # optimize parameters with sample size = 5
    optim_1 = (0.33, 0.49, 0.60, 0.44, 0.40)
    # optimize parameters with sample size = 10
    # optim_2 = (0.50, 0.41, 0.51, 0.46, 0.36)
    # optimize parameters with sample size = 20
    # optim_3 = (0.46, 0.47, 0.30, 0.43, 0.58)
    # optimize parameters with sample size = 30
    optim_4 = (0.46, 0.46, 0.53, 0.44, 0.45)
    # stacking up the data
    tt = np.vstack((optim_1, optim_4))
    for i, nsamples in enumerate(samples):
        times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)
        # evaluate the performance of the optimziation
        n, eta, s0, R, d = tt[i][0], tt[i][1], tt[i][2], tt[i][3], tt[i][4]
        print(n, eta, s0, R, d)
        post_optimize = list(map(torch.tensor, (n, eta, s0, R, d)))
        with torch.no_grad():
            eva_stress = make(*post_optimize).solve_strain(times, strains, temperatures)[:,:,0]
        # deterministic plot with optmized parameters and observed time, strain data
        ntrue = true_stresses.shape[1]
        max_true, _ = true_stresses.kthvalue(int(ntrue), dim=1)
        min_true, _ = true_stresses.kthvalue(int(1), dim=1)
        mean_true = torch.mean(true_stresses, 1)
        mean_optimize = torch.mean(eva_stress, 1)
        # plt.figure()  if plot on different figures
        plt.plot(strains[:,0].numpy(), mean_true.numpy(), lw = 3.0, ls = '--',
          label = "Actual mean with size of {}".format(nsamples))
        # plt.fill_between(strains[:,0].numpy(), min_true.numpy(), max_true.numpy(), 
          # alpha = 0.5, label = "Actual range")
        plt.plot(strains[:,0].numpy(), mean_optimize.numpy(), lw = 3.0, ls = '-',
                            label = "optimize with size of {}".format(nsamples))
        
    plt.xlabel("Strain (mm/mm)", fontsize=18)
    plt.ylabel("Stress (MPa)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
    #plt.savefig("Newton_mean_evaluation.png")
    plt.show()
    plt.close()  
  
