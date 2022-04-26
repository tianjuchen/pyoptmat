#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
  scales = [0.0,0.01,0.05,0.1,0.15]
  
  # plt.style.use('single')
  for scale in scales:
    data = xr.load_dataset("scale-%3.2f.nc" % scale)

    strain = data.strains.data[:,0]
    stress = data.stresses.data[:,0]

    plt.plot(strain, stress, lw = 2)
    plt.xlabel("Strain (mm/mm)", fontsize=16)
    plt.ylabel("Stress (MPa)", fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("visualize-%3.2f.pdf" % scale, dpi=600)
