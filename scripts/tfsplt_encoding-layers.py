import argparse
import glob
import itertools
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tfsplt_utils import (
    get_cat_color,
    get_fader_color,
    get_con_color,
    read_sig_file,
    read_folder,
)
from tfsplt_encoding import get_sigelecs, get_sid, get_elecbrain


# -----------------------------------------------------------------------------
# Argument Parser and Setup
# -----------------------------------------------------------------------------


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--sid", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", type=int, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument(
        "--sig-elec-file-dir", nargs="?", default="data/plotting/sig-elecs"
    )
    parser.add_argument("--fig-size", nargs="+", type=int, default=[18, 6])
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--x-vals-show", nargs="+", type=float, required=True)
    parser.add_argument("--lag-ticks", nargs="+", type=float, default=[])
    parser.add_argument("--lag-tick-labels", nargs="+", type=int, default=[])
    parser.add_argument("--y-vals-limit", nargs="+", type=float, default=[0, 0.3])
    parser.add_argument("--outfile", default="results/figures/tfs-encoding.pdf")
    args = parser.parse_args()

    return args


def arg_assert(args):
    """Just some sanity checks

    Args:
        args (namespace): commandline arguments

    Returns:
    """
    assert len(args.fig_size) == 2
    assert len(args.formats) == len(args.sid), "Need same number of labels as subjects"
    assert len(args.lags_show) == len(
        args.x_vals_show
    ), "Need same number of lags values and x values"
    assert all(
        lag in args.lags_plot for lag in args.lags_show
    ), "Lags plot should contain all lags from lags show"
    assert all(
        lag in args.x_vals_show for lag in args.lag_ticks
    ), "X values show should contain all values from lags ticks"
    assert all(
        lag in args.lags_show for lag in args.lag_tick_labels
    ), "Lags show should contain all values from lag tick labels"
    assert len(args.lag_ticks) == len(
        args.lag_tick_labels
    ), "Need same number of lag ticks and lag tick labels"


def get_cmap_smap(args):
    """Add line color and style map for given label key combinations to args
        cmap: dictionary of {line color: (label, key)}
        smap: dictionary of {line style: (label, key)}

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    colors = get_con_color("viridis", len(args.unique_labels))

    styles = ["-", "--", "-.", ":"]
    cmap = {}  # line color map
    smap = {}  # line style map

    for label, color in zip(args.unique_labels, colors):
        for key, style in zip(args.unique_keys, styles):
            cmap[(label, key)] = color
            smap[(label, key)] = style

    args.cmap = cmap
    args.smap = smap

    return args


def set_up_environ(args):
    """Adding necessary plotting information to args

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    args.x_vals_show = [x_val / 1000 for x_val in args.x_vals_show]
    args.lags_show = [lag / 1000 for lag in args.lags_show]
    args.lags_plot = [lag / 1000 for lag in args.lags_plot]
    args.unique_labels = list(dict.fromkeys(args.labels))
    args.unique_labels = [f"{label:02}" for label in args.unique_labels]
    args.unique_keys = list(dict.fromkeys(args.keys))

    args = get_cmap_smap(args)  # get color and style map
    args = get_sigelecs(args)  # get significant electrodes
    arg_assert(args)  # sanity checks

    return args


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


def aggregate_data(args, parallel=False):
    """Aggregate encoding data

    Args:
        args (namespace): commandline arguments

    Returns:
        df (DataFrame): df with all encoding results
    """
    data = []
    print("Aggregating data")
    for fmt in args.formats:
        for key in args.unique_keys:
            for layer in args.unique_labels:
                load_sid = get_sid(fmt, args)
                fname = fmt % (layer, key)
                data = read_folder(
                    data,
                    fname,
                    args.sigelecs,
                    (load_sid, key),
                    load_sid,
                    f"{layer:02}",
                    key,
                    "all",
                    parallel,
                )
    if not len(data):
        print("No data found")
        exit(1)
    df = pd.concat(data)
    return df


def organize_data(args, df):
    """Modify encoding data, trimming if necessary

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results

    Returns:
        df (DataFrame): df with correct columns (lags)
    """
    df.set_index(["label", "electrode", "key", "sid", "type"], inplace=True)
    assert len(args.lags_plot) == len(
        df.columns
    ), f"args.lags_plot length ({len(args.lags_plot)}) must be the same size as results ({len(df.columns)})"

    if len(args.lags_show) < len(args.lags_plot):  # plot parts of lags
        print("Trimming Data")
        chosen_lag_idx = [
            idx
            for idx, element in enumerate(args.lags_plot)
            if element in args.lags_show
        ]
        df = df.loc[:, chosen_lag_idx]  # chose from lags to show for the plot
        assert len(args.x_vals_show) == len(
            df.columns
        ), "args.lags_show length must be the same size as trimmed df column number"

    return df


def plot_average_split(args, df, pdf):
    """Plot average encoding with a split

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average plot added
    """
    print(f"Plotting Average split by key")
    fig, axes = plt.subplots(1, len(args.unique_keys), figsize=args.fig_size)
    for ax, (plot, subdf) in zip(axes, df.groupby("key", axis=0)):
        for line, subsubdf in subdf.groupby("label", axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            map_key = (line, plot)
            ax.fill_between(
                args.x_vals_show,
                vals - err,
                vals + err,
                alpha=0.2,
                color=args.cmap[map_key],
            )
            ax.plot(
                args.x_vals_show,
                vals,
                label=f"{line} ({len(subsubdf)})",
                color=args.cmap[map_key],
                ls=args.smap[map_key],
            )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(f"{plot} global average")
        # ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Aggregate data
    df = aggregate_data(args)
    df = organize_data(args, df)

    # Plotting
    print("Plotting")
    pdf = PdfPages(args.outfile)
    if len(args.y_vals_limit) == 1:  # automatic y limit
        args.y_vals_limit = [df.min().min(), df.max().max()]
    pdf = plot_average_split(args, df, pdf)

    pdf.close()

    return


if __name__ == "__main__":
    main()
