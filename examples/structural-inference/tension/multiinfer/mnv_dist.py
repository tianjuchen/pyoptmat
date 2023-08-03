import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import sys

loc = np.array([0.,-6.])
scale = np.array([[1.,-0.7],[-0.7,1.]])
truth = np.random.multivariate_normal(loc, scale,5000)

def model(data):
    loc_of_loc = torch.zeros(2)
    scale_of_loc = torch.ones(2)
    
    loc = pyro.sample("loc", dist.Normal(loc_of_loc,scale_of_loc).to_event(loc_of_loc.dim()))
    scale = pyro.param("scale",torch.eye(2,2))
    
    with pyro.iarange("count_arange", len(data)):
        pyro.sample("count", dist.MultivariateNormal(loc, scale), obs=data)


def guide(data):
    loc_of_loc_q = pyro.param("loc_of_loc_q", torch.zeros(2))
    scale_of_loc_q = pyro.param("scale_of_loc_q", torch.ones(2))
    pyro.sample("loc", dist.Normal(loc_of_loc_q,scale_of_loc_q).to_event(loc_of_loc_q.dim()))
  
  
def main(data, num_iter=2000):
    optim = Adam({"lr": 0.005})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    pyro.clear_param_store()
    start = time.time()
    losses = np.zeros(num_iter)
    for i in range(num_iter):
        losses[i] = svi.step(data)
        if i % (num_iter // 10) == 0:
            print("[iteration %04d] loss: %.4f" % (i + 1, losses[i]))
            elapsed_time = time.time()
            print("elapsed time: %.2f" % (elapsed_time - start))
    end = time.time()
    print("Loop take time %.2f" % (end - start))
    plt.plot(losses)
    plt.show()
    plt.close()


# print(torch.tensor(truth).float().shape)
# sys.exit("stop")
main(torch.tensor(truth).float(), 5000)

loc_prediction = pyro.param("loc_of_loc_q").data.numpy()
scale_tril = pyro.param("scale").data.numpy()
scale_prediction = scale_tril @ scale_tril.T
prediction = np.random.multivariate_normal(loc_prediction, scale_prediction,5000)

plt.scatter(truth[:,0],truth[:,1],c="r",label ="truth",alpha = 0.1)
plt.scatter(prediction[:,0],prediction[:,1],c="b",label ="prediction",alpha = 0.1)
plt.legend(loc = "lower left")
plt.show()
plt.close()