import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tfsplt_utils import read_folder2, read_sig_file


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--sid", type=int, required=True)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--top-dir", type=str, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)

    parser.add_argument("--has-ctx", action="store_true", default=False)
    parser.add_argument("--sig-elecs", action="store_true", default=False)

    parser.add_argument("--outfile", default="results/figures/ericplots.pdf")

    args = parser.parse_args()

    return args


def main():
    # Argparse
    args = arg_parser()
    # args = set_up_environ(args)

    print("Plotting")
    pdf = PdfPages(args.outfile)

    pdf.close()

    return


if __name__ == "__main__":
    main()
