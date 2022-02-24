import os, sys
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.easyguide import easy_guide
from pyro.optim import Adam
from torch.distributions import constraints


def model(batch, subsample, full_size):
    batch = list(batch)
    num_time_steps = len(batch)
    drift = pyro.sample("drift", dist.LogNormal(-1, 0.5))
    with pyro.plate("data", full_size, subsample=subsample):
        z = 0.
        for t in range(num_time_steps):
            z = pyro.sample("state_{}".format(t),
                            dist.Normal(z, drift))
            batch[t] = pyro.sample("obs_{}".format(t),
                                   dist.Bernoulli(logits=z),
                                   obs=batch[t])
    return torch.stack(batch)

full_size = 100
num_time_steps = 7
pyro.set_rng_seed(123456789)
data = model([None] * num_time_steps, torch.arange(full_size), full_size)
assert data.shape == (num_time_steps, full_size)

rank = 3
"""
def guide(batch, subsample, full_size):
    num_time_steps, batch_size = batch.shape

    # MAP estimate the drift.
    drift_loc = pyro.param("drift_loc", lambda: torch.tensor(0.1),
                           constraint=constraints.positive)
    pyro.sample("drift", dist.Delta(drift_loc))

    # Model local states using shared uncertainty + local mean.
    cov_diag = pyro.param("state_cov_diag",
                          lambda: torch.full((num_time_steps,), 0.01),
                         constraint=constraints.positive)
    cov_factor = pyro.param("state_cov_factor",
                            lambda: torch.randn(num_time_steps, rank) * 0.01)
    with pyro.plate("data", full_size, subsample=subsample):
        # Sample local mean.
        loc = pyro.param("state_loc",
                         lambda: torch.full((full_size, num_time_steps), 0.5),
                         event_dim=1)
        states = pyro.sample("states",
                             dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag),
                             infer={"is_auxiliary": True})
        # Unpack the joint states into one sample site per time step.
        for t in range(num_time_steps):
            pyro.sample("state_{}".format(t), dist.Delta(states[:, t]))


@easy_guide(model)
def easy_guide(self, batch, subsample, full_size):
    # MAP estimate the drift.
    self.map_estimate("drift")

    # Model local states using shared uncertainty + local mean.
    group = self.group(match="state_[0-9]*")  # Selects all local variables.
    cov_diag = pyro.param("state_cov_diag",
                          lambda: torch.full(group.event_shape, 0.01),
                          constraint=constraints.positive)
    cov_factor = pyro.param("state_cov_factor",
                            lambda: torch.randn(group.event_shape + (rank,)) * 0.01)
    with self.plate("data", full_size, subsample=subsample):
        # Sample local mean.
        loc = pyro.param("state_loc",
                         lambda: torch.full((full_size,) + group.event_shape, 0.5),
                         event_dim=1)
        # Automatically sample the joint latent, then unpack and replay model sites.
        group.sample("states", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag))
"""
@easy_guide(model)
def amortize_guide(self, batch, subsample, full_size):
    num_time_steps, batch_size = batch.shape
    self.map_estimate("drift")

    group = self.group(match="state_[0-9]*")
    cov_diag = pyro.param("state_cov_diag",
                          lambda: torch.full(group.event_shape, 0.01),
                          constraint=constraints.positive)
    cov_factor = pyro.param("state_cov_factor",
                            lambda: torch.randn(group.event_shape + (rank,)) * 0.01)

    # Predict latent propensity as an affine function of observed data.
    if not hasattr(self, "nn"):
        self.nn = torch.nn.Linear(group.event_shape.numel(), group.event_shape.numel())
        self.nn.weight.data.fill_(1.0 / num_time_steps)
        self.nn.bias.data.fill_(-0.5)
    pyro.module("state_nn", self.nn)
    with self.plate("data", full_size, subsample=subsample):
        loc = self.nn(batch.t())
        group.sample("states", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag))


def train(guide, num_epochs=101, batch_size=20):
    full_size = data.size(-1)
    pyro.get_param_store().clear()
    pyro.set_rng_seed(123456789)
    svi = SVI(model, guide, Adam({"lr": 0.02}), Trace_ELBO())
    for epoch in range(num_epochs):
        pos = 0
        losses = []
        while pos < full_size:
            subsample = torch.arange(pos, pos + batch_size)
            batch = data[:, pos:pos + batch_size]
            pos += batch_size
            losses.append(svi.step(batch, subsample, full_size=full_size))
        epoch_loss = sum(losses) / len(losses)
        if epoch % 10 == 0:
            print("epoch {} loss = {}".format(epoch, epoch_loss / data.numel()))

train(easy_guide)

print("Pyro Inferred Parameters are:")
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
print("")