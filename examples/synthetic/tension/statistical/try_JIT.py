import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO, SVI
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal, AutoDelta
from pyro.optim import Adam
from tqdm import tqdm

def model(data):
    loc = pyro.sample("loc", dist.Normal(0., 10.))
    scale = pyro.sample("scale", dist.LogNormal(0., 3.))
    with pyro.plate("data", data.size(0)):
        pyro.sample("obs", dist.Normal(loc, scale), obs=data)

guide = AutoDelta(model)

data = dist.Normal(0.5, 2.).sample((100,))

niter = 500
pyro.clear_param_store()
# elbo = Trace_ELBO()
elbo = JitTrace_ELBO()
svi = SVI(model, guide, Adam({'lr': 0.01}), elbo)
t = tqdm(range(niter), total = niter, desc = "Loss:    ")
loss_hist = []
for i in t:
    loss = svi.step(data)
    loss_hist.append(loss)
    t.set_description("Loss %3.2e" % loss)

print("Pyro Inferred Parameters are:")
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
print("")