#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
import pandas as pd
import xarray as xr
import tqdm
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    fnames = [
        "loss-historyi-0.001-0.0.txt",
        "loss-historyi-0.01-0.0.txt",
        "loss-historyi-0.1-0.0.txt",
    ]
    lrs = [0.001, 0.01, 0.1]
    
    for f, lr in zip(fnames, lrs):
        loss_history = pd.read_csv(f, header=None)
        plt.plot(
            loss_history,
            ls="-",
            lw=2.0,
            label="learning rate of {}".format(lr),
        )

    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("SVI Inference with lr=1.0x$10^{-2}$", fontsize=21)
    plt.tight_layout()
    plt.legend(prop={"size": 16}, frameon=False)
    # plt.rcParams.update({'font.size': 36})
    plt.savefig("loss-comparison.pdf")
    plt.show()
    plt.close()
