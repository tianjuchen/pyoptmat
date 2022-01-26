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



def Statistical_Param_Dist(model, scale, lr_1, lr_2, names, trues, true_std, finals, fin_std, initials, ini_std, sample_size, font):
    
    Nplot = int(len(names))
    indexs = np.arange(1, Nplot+1, 1)
    if model == 'Isotropic':
        iq = 2
    elif model == 'Chaboche':
        iq = 4
    fig = plt.figure(figsize=(Nplot*Nplot/iq,Nplot*iq))
    
    for index,name,true,true_scale,final,final_scale,prior,prior_scale in tqdm.tqdm(zip(indexs,names,trues,true_std,
        finals,fin_std,initials,ini_std), total=Nplot):
        
        if index <= Nplot/2:
            ax = fig.add_subplot(iq, Nplot/iq, index)
        else:
            ax = fig.add_subplot(iq, Nplot/iq, index)
        true_dist = dist.Normal(torch.tensor(true), torch.tensor(true_scale))
        prior_dist = dist.Normal(torch.tensor(prior), torch.tensor(prior_scale))
        posterior_dist = dist.Normal(torch.tensor(final), torch.tensor(final_scale))

        sns.distplot(true_dist.sample((100000,)), hist=True, kde_kws={"color": "r", "lw": 3, "label": "True"},
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "r"})
        sns.distplot(prior_dist.sample((100000,)), hist=True, kde_kws={"color": "b", "lw": 3, "label": "Prior"}, 
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "b"})
        sns.distplot(posterior_dist.sample((100000,)), hist=True, kde_kws={"color": "g", "lw": 3, "label": "Posterior"}, 
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "g"})

        plt.grid(False)
        # start, end = ax.get_xlim()
        # ax.xaxis.set_ticks(np.linspace(start, end, 5))
        for i, tick in enumerate(ax.xaxis.get_ticklabels()):
            if i % 2 != 0:
                tick.set_visible(False)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.ylabel("Probability",fontsize=font)
        plt.legend(prop={"size":font}, frameon=False)
        if index <= Nplot/2:
            plt.title("{} comparisons using lr-{}".format(name, lr_1), fontsize=font)
        else:
            plt.title("{} comparisons using lr-{}".format(name, lr_2), fontsize=font)
        
        plt.tight_layout()
        
    plt.savefig("SVI-feasible-{}-progress-{}-{}.png".format(model, scale, sample_size))
    return plt.close()

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


def posteriors_mean_barplot(names, scale, lr, size, data_list):
    
    iq = 0
    x = np.arange(len(names))
    width = 0.15
    interval = [x-width/2*(len(data_list)-1), x-width/2*(len(data_list)-1)/2, x,
            x+width/2*(len(data_list)-1)/2, x+width/2*(len(data_list)-1)]
    fig, ax = plt.subplots()
    for data in data_list:
        
        if iq == int(len(data_list))-1:
            ax.bar(interval[iq], (data-0.5), width)
        else:
            ax.bar(interval[iq], (data-0.5), width, label='Sample Size of {} with lr={}'.format(size[iq], lr[iq]))
        iq += 1
        
    # plt.xlim([-1, 15])
    # plt.ylim([0, 10])
    ax.set_ylabel('$\Delta \mu$', Fontsize=18)
    # ax.set_title('Variability of $\sigma^{}={}$'.format(2, scale), Fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, Fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={"size":10}, frameon=False)
    plt.grid(False)
    fig.tight_layout()
    plt.savefig("Hyparameter_on_Mean_{}.png".format(scale), dpi=600)
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


def loc_scale_evolution(names, steps, means, scales, true_means, true_scales, Data_variability, init):


    for name, mean in zip(names, means):
        
        plt.plot(steps, mean, label=name, ls='-', lw=3.0, marker='o', markersize=12)
    
    plt.plot(steps, true_means, label='True', ls='--', color='k', lw=3.0)

    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Local Evolution",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Local Evolution of Variability", fontsize=21)
    plt.grid(True)
    plt.legend(prop={"size":18}, frameon=False, ncol=3, loc='best')
    plt.tight_layout()
    # plt.savefig("Feas-Evolution of Local with Data-Variability of {} initial from {}.png".format(Data_variability, init))
    plt.show()
    plt.close()
    

    for name, scale in zip(names, scales):
        
        plt.plot(steps, scale, label=name, ls='-', lw=3.0, marker='o', markersize=12)
    
    plt.plot(steps, true_scales, label='True', ls='--', color='k', lw=3.0)

    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Scale Evolution",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Scale Evolution of Variability", fontsize=21)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(prop={"size":18}, frameon=False, ncol=3, loc='best')
    plt.tight_layout()

    # plt.savefig("Feas-Evolution of Scale with Data-Variability of {} initial from {}.png".format(Data_variability, init))
    
    return plt.show(), plt.close()



def posteriors_dist(names, true_means, true_scales, prior_means, prior_scales, post_means, 
        post_scales, vscale, lr, sz):

    for name, true_mean, true_scale, prior_mean, prior_scale, post_mean, post_scale in tqdm.tqdm(zip (names,
            true_means, true_scales, prior_means, prior_scales, post_means, post_scales),
            total=len(names)):
            
        true = dist.Normal(torch.tensor(true_mean), torch.tensor(true_scale))
        prior = dist.Normal(torch.tensor(prior_mean), torch.tensor(prior_scale))
        posterior = dist.Normal(torch.tensor(post_mean), torch.tensor(post_scale))

        sns.distplot(true.sample((100000,)), hist=True, kde_kws={"color": "r", "lw": 3, "label": "True"}, 
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "r"})
        sns.distplot(prior.sample((100000,)), hist=True, kde_kws={"color": "b", "lw": 3, "label": "Prior"}, 
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "b"})
        sns.distplot(posterior.sample((100000,)), hist=True, kde_kws={"color": "g", "lw": 3, "label": "Posterior"}, 
            hist_kws={"linewidth": 3, "alpha": 0.25, "color": "g"})


        plt.ylabel("Probability",fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("{} comparisons using SVI".format(name), fontsize=21)
        plt.grid(False)
        plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
        plt.tight_layout()
        plt.savefig("{}-progress-{}-{}-{}.png".format(name, vscale, sz, lr))
        plt.show()
        plt.close()



def deterministic_optim(names, initials, finals):
    
    for name, initial, final in zip(names, initials, finals):
        ax = plt.subplot()
        ax.axvline(0.5, color='black', linestyle='--', lw=4, label='true')
        ax.axvline(initial, color='b', linestyle='-.', lw=4, label='initial')
        ax.axvline(final, color='r', linestyle='-.', lw=4, label='final')
        ax.annotate("", xy=(final, 0.5), xytext=(initial, 0.5),
                         arrowprops=dict(arrowstyle="->", lw=10., color='g'))
                         
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)        
        plt.tight_layout()
        plt.legend(prop={"size":18}, frameon=False, ncol=1, loc='best')
        plt.grid(True)
        # plt.title("{}-LBFGS".format(name))
        plt.savefig("{}_{}_size_{}.png".format("Newton", name, 30))  #[5, 10, 20, 30]
        # plt.show()
        plt.close()


if __name__ == "__main__":



    """
    names = ["n", "eta", "s0", "R", "d"]
    vscale = [0.15]
    lr, sz = [1.0e-2, 1.0e-3], 10
    true_means = [0.50]*len(names)
    true_scales = [vscale]*len(names)
    
    prior_means = [0.31, 0.33, 0.69, 0.30, 0.38]
    prior_means += [0.06, 0.49, 0.76]
    prior_means += [0.41, 0.98, 0.21]
    prior_scales = [0.10]*len(names)
    
    post_means = [0.41, 0.51, 0.71, 0.45, 0.53]
    post_means += [0.25, 0.64, 0.78]
    post_means += [0.46, 0.86, 0.83]
    post_scales = [0.01, 0.04, 0.06, 0.03, 0.03]
    post_scales += [0.12, 0.21, 0.28]
    post_scales += [0.28, 0.13, 0.24]
    posteriors_dist(names, true_means, true_scales, prior_means, 
        prior_scales, post_means, post_scales, vscale, lr, sz)
    """
    """
    names = ["n", "eta", "s0", "R", "d"]
    variability = 0.15
    lr = [0.01, 0.01, 0.001, 0.001]
    size = [10, 20, 10, 20]
    
    #### lr = 0.01
    scale = 0.01, sample size = 10
    mean_1 = np.array([0.39, 0.71, 0.50, 0.50, 0.49])
    std_1 = np.array([0.03, 0.03, 0.02, 0.01, 0.01])
    scale = 0.01, sample size = 20
    mean_2 = np.array([0.55, 0.35, 0.60, 0.50, 0.50])
    std_2 = np.array([0.02, 0.03, 0.02, 0.01, 0.01])
    scale = 0.15, sample size = 10
    mean_3 = np.array([0.55, 0.55, 0.55, 0.51, 0.46])
    std_3 = np.array([0.08, 0.07, 0.08, 0.14, 0.07])
    # scale = 0.15, sample size = 20
    mean_4 = np.array([0.54, 0.54, 0.54, 0.55, 0.51])
    std_4 = np.array([0.07, 0.07, 0.08, 0.13, 0.13])
    """
    
    """
    #### lr = 0.001
    scale = 0.01, sample size = 10
    mean_5 = np.array([0.39, 0.71, 0.50, 0.50, 0.49])
    std_5 = np.array([0.03, 0.03, 0.02, 0.01, 0.01])
    scale = 0.01, sample size = 20
    mean_6 = np.array([0.62, 0.41, 0.44, 0.50, 0.50])
    std_6 = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    scale = 0.15, sample size = 10
    mean_7 = np.array([0.54, 0.56, 0.54, 0.51, 0.48])
    std_7 = np.array([0.07, 0.07, 0.08, 0.12, 0.05])
    scale = 0.15, sample size = 20
    mean_8 = np.array([0.54, 0.55, 0.53, 0.55, 0.51])
    std_8 = np.array([0.07, 0.06, 0.07, 0.12, 0.09])
    """
 
    # true_mean = np.array([0.50]*len(names))
    # true_std = np.array([0.15]*len(names))

    # data_list = np.vstack((mean_3,mean_4,mean_7,mean_8,true_mean))
    # std_list = np.vstack((std_3,std_4,std_7,std_8,true_std))

    # posteriors_mean_barplot(names, variability, lr, size, data_list)
    # posteriors_std_barplot(names, variability, lr, size, std_list)
    
    
    """
    names = ["n", "eta", "s0", "R", "d"]
    variability = [0.25, 0.25]
    exp_scale_levels = [0.10, 0.15]
    
    mean_1 = [0.34, 0.28, 0.43, 0.42, 0.52]
    mean_2 = [0.4538, 0.4434, 0.5129, 0.4439, 0.5335]
    mean_3 = [0.4650, 0.4806, 0.5005, 0.4363, 0.4901]
    mean_4 = [0.4673, 0.4876, 0.4941, 0.4449, 0.4674]  
    mean_5 = [0.4688, 0.4878, 0.4890, 0.4469, 0.4536]   
    mean_6 = [0.4713, 0.4896, 0.4897, 0.4509, 0.4487]   
    mean_7 = [0.47, 0.49, 0.49, 0.45, 0.45]
    
    scale_1 = [variability[1]]*len(names)
    scale_2 = [0.0685, 0.0713, 0.0782, 0.0679, 0.0859]
    scale_3 = [0.0398, 0.0427, 0.0481, 0.0415, 0.0526]
    scale_4 = [0.0283, 0.0302, 0.0352, 0.0302, 0.0371]
    scale_5 = [0.0216, 0.0235, 0.0272, 0.0237, 0.0276]
    scale_6 = [0.0173, 0.0187, 0.0221, 0.0194, 0.0223]
    scale_7 = [0.01, 0.02, 0.02, 0.02, 0.02]

    steps = np.linspace(0.0, 3000.0, 7)
    
    true_means = [0.50]*len(steps)
    true_scales = [exp_scale_levels[1]]*len(steps)
    
    means = list(np.vstack((mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7)).T)
    scales = list(np.vstack((scale_1, scale_2, scale_3, scale_4, scale_5, scale_6, scale_7)).T)
    
    loc_scale_evolution(names, steps, means, scales, true_means, true_scales, 
        exp_scale_levels[1], variability[1])
    """
    
    """
    sample_size = 10
    scale = 0.01
    lr_1, lr_2 = 1.0e-3, 1.0e-2
    model = ["Isotropic", "Chaboche"]
    names = ["n", "eta", "s0", "R", "d"]*2
    # names = ["C1", "C2", "C3", "g1", "g2", "g3"]*2
    
    trues = [0.5]*len(names)
    true_std = [scale]*len(names)
    # Need to fill
    iso_initials = [0.38, 0.65, 0.62, 0.54, 0.31,
        0.57, 0.58, 0.42, 0.62, 0.34]
    iso_finals = [0.5313, 0.6120, 0.4127, 0.4969, 0.4444,
        0.5068, 0.5417, 0.4445, 0.5003, 0.5032] 
    iso_fin_std = [0.0896, 0.0892, 0.0828, 0.0803, 0.0807,
        0.0216, 0.0094, 0.0143, 0.0103, 0.0110]
    
    
    ###############
    ini_std = [0.10]*len(names)
    fontsize = 40
    Statistical_Param_Dist(model[0], scale, lr_1, lr_2, names, trues, true_std, 
        iso_finals, iso_fin_std, iso_initials, 
        ini_std, sample_size, fontsize)
    """
    
    """
    scale = [0.01, 0.15]
    names = ['n', 'eta', 's0', 'R', 'd', 'C1', 'C2', 'C3', 'g1', 'g2', 'g3']
    size = [10, 20, 10, 20]
    lr = [1e-3, 1e-3, 1e-2, 1e-2]
    
    #### lr = 0.001
    # scale = 0.01, sample size = 10
    mean_1 = np.array([0.41, 0.51, 0.71, 0.45, 0.53, 0.25, 0.64, 0.78, 0.46, 0.86, 0.83])
    std_1 = np.array([0.01, 0.04, 0.06, 0.03, 0.03, 0.12, 0.21, 0.28, 0.28, 0.13, 0.24])
    # scale = 0.01, sample size = 20
    mean_2 = np.array([0.57, 0.43, 0.50, 0.51, 0.48, 0.43, 0.58, 0.35, 0.79, 0.92, 0.13])
    std_2 = np.array([0.02, 0.04, 0.05, 0.01, 0.01, 0.16, 0.18, 0.19, 0.21, 0.11, 0.32])
    # scale = 0.15, sample size = 10
    mean_3 = np.array([0.52, 0.47, 0.42, 0.53, 0.58, 0.85, 0.06, 0.26, 0.39, 0.95, 0.56])
    std_3 = np.array([0.10, 0.13, 0.10, 0.12, 0.12, 0.17, 0.11, 0.22, 0.23, 0.16, 0.29])
    # scale = 0.15, sample size = 20
    mean_4 = np.array([0.38, 0.59, 0.55, 0.46, 0.59, 0.71, 0.40, 0.55, 0.42, 0.40, 0.15])
    std_4 = np.array([0.11, 0.12, 0.13, 0.11, 0.11, 0.19, 0.24, 0.25, 0.26, 0.26, 0.22])
    
    
    #### lr = 0.01
    # scale = 0.01, sample size = 10
    mean_5 = np.array([0.46, 0.54, 0.53, 0.58, 0.43, 0.66, 0.35, 0.10, 0.32, 0.42, -0.04])
    std_5 = np.array([0.02, 0.03, 0.03, 0.05, 0.04, 0.15, 0.15, 0.17, 0.21, 0.26, 0.02])
    # scale = 0.01, sample size = 20
    mean_6 = np.array([0.51, 0.50, 0.48, 0.39, 0.66, 0.55, 0.42, 0.66, 0.48, 0.55, 0.32])
    std_6 = np.array([0.01, 0.01, 0.02, 0.06, 0.07, 0.03, 0.03, 0.05, 0.07, 0.12, 0.35])
    # scale = 0.15, sample size = 10
    mean_7 = np.array([0.39, 0.54, 0.61, 0.47, 0.55, 0.62, 0.25, 0.44, 0.42, 1.00, 1.07])
    std_7 = np.array([0.12, 0.13, 0.17, 0.13, 0.11, 0.15, 0.16, 0.24, 0.20, 0.14, 0.05])
    # scale = 0.15, sample size = 20
    mean_8 = np.array([0.48, 0.49, 0.51, 0.55, 0.48, 0.58, 0.42, -0.02, 0.37, 0.75, 0.67])
    std_8 = np.array([0.12, 0.15, 0.13, 0.13, 0.11, 0.19, 0.20, 0.13, 0.24, 0.19, 0.47])
    
    true_mean = np.array([0.50]*len(names))
    true_std = np.array([0.15]*len(names))
    
    
    data_list = np.vstack((mean_3,mean_4,mean_7,mean_8,true_mean))
    posteriors_mean_barplot(names, scale[1], lr, size, data_list)
    """

    """
    sample_size = 20
    scale = 0.01
    lr_1, lr_2 = 1.0e-3, 1.0e-2
    model = ["Isotropic", "Chaboche"]
    names = ["n", "eta", "s0", "R", "d"]*2
    # names = ["C1", "C2", "C3", "g1", "g2", "g3"]*2
    
    trues = [0.5]*len(names)
    true_std = [scale]*len(names)
    # Need to fill
    iso_initials = [0.71, 0.46, 0.70, 0.69, 0.66,
        0.35, 0.38, 0.29, 0.29, 0.66]
    iso_finals = [0.57, 0.43, 0.50, 0.51, 0.48,
        0.51, 0.50, 0.48, 0.39, 0.66] 
    iso_fin_std = [0.02, 0.04, 0.05, 0.01, 0.01,
        0.01, 0.01, 0.02, 0.06, 0.07]
    
    chabo_initials = [0.78, 0.93, 0.74, 0.39, 0.65, 0.00,
        0.43, 0.29, 0.49, 0.73, 0.83, 0.26]
    chabo_finals = [0.43, 0.58, 0.35, 0.79, 0.92, 0.13,
        0.55, 0.42, 0.66, 0.48, 0.55, 0.32]
    chabo_fin_std = [0.16, 0.18, 0.19, 0.21, 0.11, 0.32,
        0.03, 0.03, 0.05, 0.07, 0.12, 0.35]
    
    ###############
    ini_std = [0.10]*len(names)
    fontsize = 40
    Statistical_Param_Dist(model[0], scale, lr_1, lr_2, names, trues, true_std, 
        iso_finals, iso_fin_std, iso_initials, 
        ini_std, sample_size, fontsize)
    """