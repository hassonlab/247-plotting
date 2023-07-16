import argparse
import glob
import itertools
import os
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tfsplt_utils import (
    get_fader_color,
    get_con_color,
    read_folder,
)
from tfsplt_encoding import get_sigelecs, get_elecbrain


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
    parser.add_argument("--colors", nargs="?", type=str, default="viridis")
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
    assert len(args.formats) == 1, "Need exactly 1 format"
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
    colors = get_con_color(args.colors, len(args.unique_labels))

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
    args.formats = args.formats[0]

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
    for load_sid in args.sid:
        for key in args.unique_keys:
            for layer in args.unique_labels:
                fname = args.formats % (load_sid, layer, key)
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


def plot_average_encoding(args, df, pdf):
    """Plot average encoding split by keys

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average plot added
    """
    print(f"Plotting Average Encoding split by key")
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
                color=args.cmap[map_key],
                ls=args.smap[map_key],
            )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(f"{plot} average ({len(subsubdf)})")
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")

    norm = mpl.colors.Normalize(vmin=args.labels[0], vmax=args.labels[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=args.colors)
    ticks = np.linspace(args.labels[0], args.labels[-1], 4).astype(int)
    plt.colorbar(
        sm,
        ax=fig.get_axes(),
        fraction=0.03,
        pad=0.02,
        ticks=ticks,
        label="Layers",
    )

    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_encoding_heatmap(args, df, pdf):
    """Plot average encoding heatmap split by keys

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average heatmap plot added
    """
    print(f"Plotting Average Encoding Heatmap split by key")
    xticks = np.linspace(0, len(args.x_vals_show) - 1, 5)
    xticklabels = np.round(np.linspace(args.x_vals_show[0], args.x_vals_show[-1], 5), 1)
    df.columns = args.x_vals_show
    fig, axes = plt.subplots(1, len(args.unique_keys), figsize=args.fig_size)
    for ax, (plot, subdf) in zip(axes, df.groupby("key", axis=0)):
        heatmapdf = subdf.groupby("label").mean()
        sns.heatmap(heatmapdf, cmap=args.colors, linewidths=0, rasterized=True, ax=ax)
        ax.invert_yaxis()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Layers")
        ax.set_title(f"{plot} average")

    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_max(args, df, pdf):
    """Plot average scatter plot with encoding max split by keys

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average plot added
    """
    print(f"Plotting Average Max Correlation split by key")
    colors = get_con_color(args.colors, len(args.unique_labels))
    fig, axes = plt.subplots(1, len(args.unique_keys), figsize=args.fig_size)
    for ax, (plot, subdf) in zip(axes, df.groupby("key", axis=0)):
        scatterdf = subdf.groupby("label").mean().max(axis=1)
        ax.scatter(
            args.labels,
            scatterdf,
            s=70,
            marker="o",
            color=colors,
            # facecolors="none",
            # edgecolors="grey",
        )
        ax.set(
            xlabel="Layers",
            ylabel="Max Correlation (r)",
            title=f"{plot} average max",
        )

    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes_encoding(args, df, pdf):
    """Plot individual electrodes encoding split by keys

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct encoding plots added
    """
    print(f"Plotting Individual Elecs Encoding split by keys")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, axes = plt.subplots(1, len(args.unique_keys), figsize=args.fig_size)
        for _, (key, subsubdf) in zip(axes, subdf.groupby("key")):
            ax = axes[args.unique_keys.index(key)]
            for row, values in subsubdf.iterrows():
                layer = row[0]
                map_key = (layer, key)
                ax.plot(
                    args.x_vals_show,
                    values,
                    color=args.cmap[map_key],
                    ls=args.smap[map_key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.set_ylim(args.y_vals_limit[0] - 0.05, args.y_vals_limit[1] + 0.05)
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {key}",
            )

        pdf.savefig(fig)
        plt.close()
    return pdf


def plot_maxidx(args, df, pdf):
    """Plot elec max_idx split by keys

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct scatterplot added
    """
    print(f"Plotting Max Lag split by keys")
    colors = get_con_color(args.colors, len(args.unique_labels))
    fig, axes = plt.subplots(1, len(args.unique_keys), figsize=args.fig_size)
    for ax, (plot, subdf) in zip(axes, df.groupby("key", axis=0)):
        for line, subsubdf in subdf.groupby("label", axis=0):
            scatterdf = subsubdf.idxmax(axis=1)
            ax.scatter(
                scatterdf,
                [int(line)] * len(scatterdf),
                s=20,
                marker="o",
                color=colors[int(line)],
            )
            ax.set(
                xlabel="Lags (s)",
                ylabel="Layers",
                title=f"{plot} max_idx",
            )
            ax.set_xlim(args.lags_show[0], args.lags_show[-1])
            ax.set_ylim(-0.25, int(args.labels[-1]) + 0.25)
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")

    norm = mpl.colors.Normalize(vmin=args.labels[0], vmax=args.labels[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=args.colors)
    ticks = np.linspace(args.labels[0], args.labels[-1], 4).astype(int)
    plt.colorbar(
        sm,
        ax=fig.get_axes(),
        fraction=0.03,
        pad=0.02,
        ticks=ticks,
        label="Layers",
    )

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
    pdf = PdfPages(args.outfile)
    if len(args.y_vals_limit) == 1:  # automatic y limit
        args.y_vals_limit = [df.min().min(), df.max().max()]
    pdf = plot_average_encoding(args, df, pdf)
    pdf = plot_average_encoding_heatmap(args, df, pdf)
    pdf = plot_average_max(args, df, pdf)
    # pdf = plot_electrodes_encoding(args, df, pdf)
    # pdf = plot_maxidx(args, df, pdf)

    pdf.close()

    return


if __name__ == "__main__":
    main()
