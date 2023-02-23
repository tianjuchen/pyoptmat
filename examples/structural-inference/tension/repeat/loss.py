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

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/"
    path2 = "examples/structural-inference/tension/repeat/"
    path = path1 + path2

    fnames = glob.glob(path + "*.txt")

    for f in fnames:
        fn = os.path.basename(f).split(".txt")[0]
        
        if fn == "pyoptmat-history":
            loss_history = pd.read_csv(f)
            
            print(f)
            
            plt.plot(
                loss_history,
                ls="-",
                lw=3.0,
                # label="learning rate of {}".format(lr),
            )

            ax = plt.gca()
            # plt.yscale("log")
            plt.xlabel("Step", fontsize=18)
            plt.ylabel("Loss", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.locator_params(axis='both', nbins=4)
            # plt.title("SVI Inference with lr=1.0x$10^{-2}$", fontsize=21)
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(3)
            # increase tick width
            ax.tick_params(width=3)
            plt.tight_layout()
            # plt.legend(prop={"size": 18}, frameon=False)
            # plt.rcParams.update({'font.size': 36})
            # plt.savefig("loss-comparison.pdf")
            plt.show()
            plt.close()
        