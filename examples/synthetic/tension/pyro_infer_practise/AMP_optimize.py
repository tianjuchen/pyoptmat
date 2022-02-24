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

print(torch.__version__)

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# Run on CPU (home machine GPU is eh)
# dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, use_adjoint = True,
      device = device, **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.15
  nsamples = 5 # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)

  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d"]
  # ics = [ra.uniform(0,1) for i in range(len(names))]
  ics = [0.67, 0.09, 0.53, 0.80, 0.53]

  print("Initial parameter values:")
  for n, ic in zip(names, ics):
    print("%s:\t%3.2f" % (n, ic))
  print("")
  
  # 3) Create the actual model
  model = optimize.DeterministicModel(make, names, ics)

  # 4) Setup the optimizer
  niter = 200
  lr = 1.0e-2
  optim = torch.optim.Adam(model.parameters(), lr = lr)

  # 5) Setup the objective function
  loss = torch.nn.MSELoss(reduction = 'sum')

  # 6) Actually do the optimization!
  scaler = torch.cuda.amp.GradScaler()
  t = tqdm(range(niter), total = niter, desc = "Loss:    ")
  loss_history = []
  for i in t:
    optim.zero_grad()
    with torch.cuda.amp.autocast():
      pred = model(times, strains, temperatures)
      lossv = loss(pred, true_stresses)
    scaler.scale(lossv).backward()
    scaler.step(optim)
    scaler.update()
    loss_history.append(lossv.detach().cpu().numpy())
    t.set_description("Loss: %3.2e" % loss_history[-1])
  
  # 7) Check accuracy of the optimized parameters
  print("")
  print("Optimized parameter accuracy:")
  for n in names:
    print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

  # 8) Save the convergence history
  np.savetxt("loss-history.txt", loss_history)

  plt.figure()
  plt.plot(loss_history)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig("convergence.pdf")
