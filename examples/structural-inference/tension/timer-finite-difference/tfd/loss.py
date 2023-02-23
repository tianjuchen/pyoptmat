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


def load_txt(path, fn):
    fnames = glob.glob(path + "*.txt")
    for f in fnames:
        name = os.path.basename(f).split(".txt")[0]
        if name == fn:
            df = pd.read_csv(f)
            return df


if __name__ == "__main__":

    path1 = "/mnt/c/Users/ladmin/Desktop/argonne/old_pyoptmat/pyoptmat/examples/structural-inference/"
    path2 = "tension/timer-finite-difference/tfd/"

    path = path1 + path2

    fn_lbfgs_loss = load_txt(path, "bfgs-2point-history")
    fn_lbfgs_time = load_txt(path, "bfgs-2point-timer")

    # fn_lbfgs_loss = load_txt(path, "Nelder-Mead-history")
    # fn_lbfgs_time = load_txt(path, "Nelder-Mead-timer")

    plt.plot(fn_lbfgs_time, fn_lbfgs_loss)
    plt.show()
    plt.close()
