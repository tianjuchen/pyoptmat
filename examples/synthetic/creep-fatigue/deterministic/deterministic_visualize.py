#!/usr/bin/env python3

import sys
sys.path.append('../../../..')
sys.path.append('..')

import os, glob
import numpy as np
import numpy.random as ra
import pandas as pd
import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize
from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)

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
def make(n, eta, s0, R, d, C, g, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, C, g, 
      device = device, use_adjoint = True,**kwargs).to(device)


def fetch_data(path, file_name):
  fnames = glob.glob(path + file_name + '*.txt')
  for f in fnames:
    df = pd.read_csv(f, header=None)
  return df

if __name__ == "__main__":

 
  scale = 0.15
  nsamples = 30 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)

  path = '/mnt/c/Users/ladmin/Desktop/argonne/20210624/cyclic_deterministic_optimize/\
paper_1_cyclic_updates/synthetic_fatigue/single_batch/adam/'

  names = ["n", "eta", "s0", "R", "d", "C", "g"]
  n, eta, s0, R, d, C, g = map(torch.tensor, (0.29, 0.63, 0.73, 0.51, 0.54,
                               [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]))
    
  model = make(n, eta, s0, R, d, C, g)
  with torch.no_grad():
    pred_stress = model.solve_strain(times, strains, temperatures)[:,:,0]
    
  plt.plot(strains[:,10], true_stresses[:,10], 'c', label = "Actual")
  plt.plot(strains[:,10], pred_stress[:,10], color='orange', ls='--', label = "Optimized")
  plt.xlabel("Strain (mm/mm)",fontsize=18)
  plt.ylabel("Stress (MPa)",fontsize=18)
  plt.xticks(fontsize=11)
  plt.yticks(fontsize=11)
  plt.grid(True)
  plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
  # plt.savefig(path + "single_batch_{}.png".format("Adam"))
  plt.show()
  plt.close()    

  
  """
  names = ["n", "eta", "s0", "R", "d", "C", "g"]
  path_1 = '/mnt/c/Users/ladmin/Desktop/argonne/synthetic_fatigue/test_initials/random_initials/'
  path_2 = '/mnt/c/Users/ladmin/Desktop/argonne/synthetic_fatigue/test_initials/same_initials/'
  fn_1 = 'grad_norm-test-1'
  fn_2 = 'grad_norm-1'
  
  df_1 = fetch_data(path_1, fn_1)
  df_2 = fetch_data(path_2, fn_2)
  
  indices = np.arange(6, len(df_1), len(names))
  data_1 = df_1.take(indices)
  data_2 = df_2.take(indices)

  # plt.plot(data_1, 'r', label = "random initial")
  plt.plot(data_2, 'g--', label = "same initial")
  plt.xlabel("Step")
  plt.ylabel("Grad_Norm")
  plt.legend(loc='best')
  plt.grid(True)
  # plt.savefig("optimized_{}.pdf".format(mode))
  plt.show()
  plt.close()  
  """  