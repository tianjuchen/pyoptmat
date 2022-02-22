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
  scale = 0.15
  nsamples = 5 #[5, 10, 20, 30] # at each strain rate
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)
  
  
  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d"]
  ics = [ra.uniform(0,1) for i in range(len(names))]

  print("Initial parameter values:")
  for n, ic in zip(names, ics):
    print("%s:\t%3.2f" % (n, ic))
  print("")
  
  # 3) Create the actual model
  model = optimize.DeterministicModel(make, names, ics)

  # 4) Setup the optimizer
  niter = 1
  optim = torch.optim.LBFGS(model.parameters())

  # 5) Setup the objective function
  loss = torch.nn.MSELoss(reduction = 'sum')

  # 6) Actually do the optimization!
  def closure():
    optim.zero_grad()
    pred = model(times, strains, temperatures)
    lossv = loss(pred, true_stresses)
    lossv.backward()
    return lossv

  t = tqdm(range(niter), total = niter, desc = "Loss:    ")
  loss_history = []
  for i in t:
    closs = optim.step(closure)
    loss_history.append(closs.detach().cpu().numpy())
    t.set_description("Loss: %3.2e" % loss_history[-1])
  
  # 7) Check accuracy of the optimized parameters
  print("")
  print("Optimized parameter accuracy:")
  post_optimize = []
  for n in names:
    print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))
    post_optimize.append(getattr(model, n).data)
    

  print(post_optimize)
  # 8) Save the convergence history
  np.savetxt("loss-{}-{}.txt".format(scale, nsamples), loss_history)

  plt.figure()
  plt.plot(loss_history)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.grid(True)
  plt.savefig("convergence-{}-{}.pdf".format(scale, nsamples))
  plt.show()
  plt.close()    
  
  # 9) evaluate the performance of the optimziation
  with torch.no_grad():
    eva_stress = make(*post_optimize).solve_strain(times, strains, temperatures)[:,:,0]
  ## deterministic plot with optmized parameters and random time, strain data
  print(eva_stress.size())
  plt.figure()
  plt.plot(strains[:,0].numpy(), eva_stress[:,0].numpy(), 'g-', label = 'optimize')
  plt.plot(strains[:,0].numpy(), true_stresses[:,0].numpy(), 'b--', label = 'data')
  plt.xlabel("Strain (mm/mm)")
  plt.ylabel("Stress (MPa)")
  plt.tight_layout()
  plt.legend()
  plt.grid(True)
  # plt.savefig("Deterministic-evaluation-{}-{}.png".format(scale, nsamples))
  plt.show()
  plt.close()  
  
  
  # ) 10 evaluate the deterministic performance
  ## deterministic plot with optmized parameters and observed time, strain data
  ntrue = true_stresses.shape[1]
  max_true, _ = true_stresses.kthvalue(int(ntrue), dim=1)
  min_true, _ = true_stresses.kthvalue(int(1), dim=1)
  mean_true = torch.mean(true_stresses, 1)
  mean_optimize = torch.mean(eva_stress, 1)
  plt.figure()
  plt.plot(strains[:,0].numpy(), mean_true.numpy(), 'k-', 
      label = "Actual mean")
  # plt.fill_between(strains[:,0].numpy(), min_true.numpy(), max_true.numpy(), 
      # alpha = 0.5, label = "Actual range")
  plt.plot(strains[:,0].numpy(), mean_optimize.numpy(), 'g-', label = 'optimize')
  plt.xlabel("Strain (mm/mm)")
  plt.ylabel("Stress (MPa)")
  plt.tight_layout()
  plt.legend()
  plt.grid(True)
  plt.savefig("Range-location-evaluation-{}-{}.png".format(scale, nsamples))
  plt.show()
  plt.close()  
  