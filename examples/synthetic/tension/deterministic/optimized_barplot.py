#!/usr/bin/env python3

import sys
import os, glob
import numpy as np
import numpy.random as ra

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tqdm
import csv
import scipy.interpolate as inter
from scipy.stats import norm, beta
import torch, seaborn as sns
import pyro.distributions as dist
import warnings
warnings.filterwarnings("ignore")





def loss_history_visualize(paths, labels, colors, markers, optimizer, filename):
    loss_history = []
    for path in paths:
        for files in glob.glob(path +"*.txt"):
         loss_history_temp = pd.read_csv(files, header=None)
         loss_history.append(loss_history_temp)
    for i in range(len(loss_history)):
        # plt.plot(loss_history[i], label=labels[i], ls='-', lw=3.0, color=colors[i])
        
        if i < len(loss_history)/2.0:
            # continue
            plt.plot(loss_history[i], label=labels[i], ls='-', lw=3.0, color=colors[i]) #, marker=markers[i], markersize=12)
        else:
            # continue
            plt.plot(loss_history[i], label=labels[i], ls='--', lw=3.0, 
                color=colors[i-int(len(loss_history)/2.0)]) #, marker=markers[i-int(len(loss_history)/2.0)], markersize=12)
        
    # plt.figure()
    plt.xlim([-1, 10])
    # plt.ylim([0, 10000])
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Loss",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("{}".format(filename), fontsize=21)
    plt.tight_layout()
    plt.grid(False)
    plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
    # plt.rcParams.update({'font.size': 36})
    # plt.savefig("{}-{}.png".format(filename, Cyclic), dpi=600)
    return plt.show(), plt.close()


def posteriors_mean_barplot(names, size, data_list):
    
    iq = 0
    x = np.arange(len(names))
    width = 0.15
    interval = [x-width/2*(len(data_list)-1), x-width/2*(len(data_list)-1)/2, x,
            x+width/2*(len(data_list)-1)/2, x+width/2*(len(data_list)-1)]
    fig, ax = plt.subplots()
    for data in data_list:
        
        if iq == int(len(data_list))-1:
            ax.bar(interval[iq], abs(data-0.5), width)
        else:
            ax.bar(interval[iq], abs(data-0.5), width, label='Sample Size = {}'.format(size[iq]))
        iq += 1
        
    ax.set_ylabel('Optimized Gap', Fontsize=18)
    # ax.set_title('Variability of $\sigma^{}={}$'.format(2, scale), Fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, Fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={"size":12}, frameon=False)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig("Optimized_barplot_Newton.png")
    return plt.show(), plt.close()
    
def posteriors_std_barplot(names, scale, lr, size, data_list):
    
    iq = 0
    x = np.arange(len(names))
    width = 0.15
    interval = [x-width/2*(len(data_list)-1), x-width/2*(len(data_list)-1)/2, x,
            x+width/2*(len(data_list)-1)/2, x+width/2*(len(data_list)-1)]
    fig, ax = plt.subplots()
    for data in data_list:
        
        if iq == int(len(data_list))-1:
            ax.bar(interval[iq], (data-0.15), width, label='True')
        else:
            ax.bar(interval[iq], (data-0.15), width, label='Sample Size of {} with lr={}'.format(size[iq], lr[iq]))
        iq += 1
        
    # plt.xlim([-1, 15])
    # plt.ylim([0, 10])
    ax.set_ylabel('$\Delta \sigma^{2}$', Fontsize=18)
    # ax.set_title('Variability of $\sigma^{}={}$'.format(2, scale), Fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, Fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={"size":10}, frameon=False)
    plt.grid(False)
    fig.tight_layout()
    plt.savefig("Hyparameter_on_Variance_{}.png".format(scale), dpi=600)
    return plt.show(), plt.close()

#================================================#
def load_file(path, size, method):
#================================================#
  fnames = glob.glob(path + "*.txt")
  for f in fnames:
    optimizer = os.path.basename(f).split('_')[0]
    if optimizer == method:
      sample_size = os.path.basename(f).split('_')[1].split('.')[0]
    if optimizer == method and sample_size == str(size):
      df = pd.read_csv(f, usecols=[0], names=['Loss'], header=None)
      return df


if __name__ == "__main__":


    """
    # barplot of the discrepancy between true and optimized values
    names = ["n", "eta", "s0", "R", "d"]
    variability = 0.15
    size = [5, 10, 20, 30]
    mean_1 = np.array([0.33, 0.49, 0.60, 0.44, 0.40])
    mean_2 = np.array([0.43, 0.43, 0.60, 0.43, 0.42])
    mean_3 = np.array([0.47, 0.44, 0.55, 0.42, 0.42])
    mean_4 = np.array([0.46, 0.46, 0.53, 0.44, 0.45])
    true_mean = np.array([0.50]*len(names))
    data_list = np.vstack((mean_1,mean_2,mean_3,mean_4,true_mean))
    posteriors_mean_barplot(names, size, data_list)    
    """
    
    path_1 = "/mnt/c/Users/ladmin/Desktop/argonne/JIT-pyoptmat/pyoptmat/examples/synthetic/tension/deterministic/"
    methods = ["adam", "newton"]
    sizes = [5, 30]
    for method in methods:
      for size in sizes:
        df = load_file(path_1, size, method)
        plt.plot(df['Loss']/size, lw=3.0, label='Size = {}, Optim = {}'.format(size, method))
    
    plt.xlim([-1, 50])
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Loss",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
    plt.savefig("Loss_evolution.png")
    plt.show()
    plt.close()
    
