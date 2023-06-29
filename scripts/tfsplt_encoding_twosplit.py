import glob
import argparse
import os
import pandas as pd
import itertools
import numpy as np
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tfsplt_utils import (
    get_cat_color,
    get_fader_color,
    get_con_color,
    read_sig_file,
    read_folder,
)
from tfsplt_encoding import get_elecbrain


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
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--fig-size", nargs="+", type=int, default=[18, 6])
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument(
        "--sig-elec-file-dir", nargs="?", default="data/plotting/sig-elecs"
    )
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--x-vals-show", nargs="+", type=float, required=True)
    parser.add_argument("--lag-ticks", nargs="+", type=float, default=[])
    parser.add_argument("--lag-tick-labels", nargs="+", type=int, default=[])
    parser.add_argument(
        "--y-vals-limit", nargs="+", type=float, default=[0, 0.3]
    )
    parser.add_argument("--lc-by", type=int, default=0)
    parser.add_argument("--ls-by", type=int, default=1)
    parser.add_argument("--split-hor", type=int, default=1)
    parser.add_argument("--split-ver", type=int, default=2)
    parser.add_argument("--outfile", default="results/figures/tmp.pdf")
    args = parser.parse_args()
    return args


def arg_assert(args):
    """Just some sanity checks

    Args:
        args (namespace): commandline arguments

    Returns:
    """
    assert len(args.fig_size) == 2
    assert len(args.formats) == len(
        args.keys
    ), "Need same number of labels as formats"
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

    return args


def get_cmap_smap(args):
    """Add line color and style map for given label key combinations to args
        cmap: dictionary of {line color: (label, key)}
        smap: dictionary of {line style: (label, key)}

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    if len(args.unique_keys[args.lc_by]) > 10:
        colors = get_con_color("viridis", len(args.unique_labels))
    else:
        colors = get_cat_color()

    styles = ["-", "--", "-.", ":"]
    cmap = {}  # line color map
    smap = {}  # line style map

    for lc, color in zip(args.unique_keys[args.lc_by], colors):
        for ls, style in zip(args.unique_keys[args.ls_by], styles):
            for source in args.keys:
                if source[args.lc_by] == lc and source[args.ls_by] == ls:
                    cmap[source] = color
                    smap[source] = style
    args.cmap = cmap
    args.smap = smap
    return args


def get_sigelecs(args):
    """Add significant electrode lists to args
        sigelecs (Dict): Dictionary in the following format
            tuple of (sid,key) : [list of sig elec]

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    sigelecs = {}
    if len(args.sig_elec_file) == 0:
        pass
    elif len(args.sig_elec_file) == len(args.sid) * len(args.unique_keys[1]):
        sid_key_tup = [
            x for x in itertools.product(args.sid, args.unique_keys[1])
        ]
        for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
            sigelecs[sid_key] = read_sig_file(fname, args.sig_elec_file_dir)
    else:
        raise Exception(
            "Need a significant electrode file for each subject-key combo"
        )
    args.sigelecs = sigelecs
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

    args.keys = [tuple(i.split()) for i in args.keys]
    keys = np.array(args.keys)
    unique_keys = []
    for i in np.arange(0, 3):
        _, idx = np.unique(keys[:, i], return_index=True)
        unique_keys.append(keys[np.sort(idx), i].tolist())
    args.unique_keys = unique_keys

    args = get_cmap_smap(args)  # get color and style map
    args = get_sigelecs(args)  # get significant electrodes
    arg_assert(args)  # some sanity checks

    return args


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


def add_sid(df, elec_name_dict):
    elec_name = df.index.to_series().str.get(1).tolist()
    sid_name = df.index.to_series().str.get(3).tolist()
    for idx, string in enumerate(elec_name):
        if string.find("_") < 0 or not string[0:3].isdigit():  # no sid in front
            new_string = str(sid_name[idx]) + "_" + string  # add sid
            elec_name_dict[string] = new_string
        else:
            elec_name_dict[string] = string
    return elec_name_dict


# -----------------------------------------------------------------------------
# Aggregate Data Functions
# -----------------------------------------------------------------------------


def aggregate_data(args, parallel=False):
    data = []
    print("Aggregating data")
    for fmt, label in zip(args.formats, args.keys):
        load_sid = 0
        for sid in args.sid:
            if str(sid) in fmt:
                load_sid = sid
        assert (
            load_sid != 0
        ), f"Need subject id for format {fmt}"  # check subject id for format is provided
        fname = fmt % label[1]
        data = read_folder(
            data,
            fname,
            args.sigelecs,
            (load_sid, label[1]),
            load_sid,
            label[0],
            label[1],
            label[2],
            parallel,
        )
    if not len(data):
        print("No data found")
        exit(1)
    df = pd.concat(data)
    return df


def organize_data(args, df):
    df.set_index(
        ["sid", "electrode", "label", "mode", "type"],
        inplace=True,
    )

    n_lags, n_df = len(args.lags_plot), len(df.columns)
    assert (
        n_lags == n_df
    ), "args.lags_plot length ({n_av}) must be the same size as results ({n_df})"

    elec_name_dict = {}
    # new_sid = df.index.to_series().str.get(1).apply(add_sid) # add sid if no sid in front
    elec_name_dict = add_sid(df, elec_name_dict)
    df = df.rename(
        index=elec_name_dict
    )  # rename electrodes to add sid in front

    if len(args.lags_show) < len(args.lags_plot):  # plot parts of lags
        print("Trimming Data")
        lags_plot = [lag / 1000 for lag in args.lags_plot]
        chosen_lag_idx = [
            idx
            for idx, element in enumerate(lags_plot)
            if element in args.lags_show
        ]
        df = df.loc[:, chosen_lag_idx]  # chose from lags to show for the plot
        assert len(args.x_vals_show) == len(
            df.columns
        ), "args.lags_show length must be the same size as trimmed df column number"

    return df


# -----------------------------------------------------------------------------
# Plotting Average and Individual Electrodes
# -----------------------------------------------------------------------------


def plot_average_split_by_key(args, df, pdf):
    fig, axes = plt.subplots(
        len(args.unique_keys[args.split_ver]),
        len(args.unique_keys[args.split_hor]),
        figsize=(args.fig_size[0], args.fig_size[1]),
    )
    for i, split_ver_key in enumerate(args.unique_keys[args.split_ver]):
        for j, split_hor_key in enumerate(args.unique_keys[args.split_hor]):
            ax = axes[i, j]
            idx = pd.IndexSlice
            if args.split_ver == 2:
                subdf = df.loc[
                    idx[:, :, :, split_hor_key, split_ver_key], :
                ]  # need to modify
            elif args.split_ver == 0:
                subdf = df.loc[
                    idx[:, :, split_ver_key, split_hor_key, :], :
                ]  # need to modify
            for label, subsubdf in subdf.groupby("label", axis=0):
                vals = subsubdf.mean(axis=0)
                err = subsubdf.sem(axis=0)
                key = (
                    subsubdf.index.get_level_values(2)[0],
                    subsubdf.index.get_level_values(3)[0],
                    subsubdf.index.get_level_values(4)[0],
                )
                ax.fill_between(
                    args.x_vals_show,
                    vals - err,
                    vals + err,
                    alpha=0.2,
                    color=args.cmap[key],
                )
                ax.plot(
                    args.x_vals_show,
                    vals,
                    label=f"{label} ({len(subsubdf)})",
                    color=args.cmap[key],
                    ls=args.smap[key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            if i == 0:
                ax.set_title(key[args.split_hor] + " global average")
            if j == 0:
                ax.set_ylabel(key[args.split_ver])
            ax.set_ylim(-0.005, 0.125)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper right", frameon=False)
            # ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes_split_by_key(args, df, pdf):
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, axes = plt.subplots(
            len(args.unique_keys[args.split_ver]),
            len(args.unique_keys[args.split_hor]),
            figsize=(args.fig_size[0], args.fig_size[1]),
        )
        for i, split_ver_key in enumerate(args.unique_keys[args.split_ver]):
            for j, split_hor_key in enumerate(args.unique_keys[args.split_hor]):
                ax = axes[i, j]
                idx = pd.IndexSlice
                try:
                    if args.split_ver == 2:
                        subsubdf = subdf.loc[
                            idx[:, :, :, split_hor_key, split_ver_key], :
                        ]  # need to modify
                    elif args.split_ver == 0:
                        subsubdf = subdf.loc[
                            idx[:, :, split_ver_key, split_hor_key, :], :
                        ]  # need to modify
                except:
                    continue
                for row, values in subsubdf.iterrows():
                    key = (row[2], row[3], row[4])
                    ax.plot(
                        args.x_vals_show,
                        values,
                        label=row[2],
                        color=args.cmap[key],
                        ls=args.smap[key],
                    )
                if len(args.lag_ticks) != 0:
                    ax.set_xticks(args.lag_ticks)
                    ax.set_xticklabels(args.lag_tick_labels)
                if i == 0:
                    ax.set_title(f"{sid} {electrode} {key[args.split_hor]}")
                if j == 0:
                    ax.set_ylabel(key[args.split_ver])
                ax.axhline(0, ls="dashed", alpha=0.3, c="k")
                ax.axvline(0, ls="dashed", alpha=0.3, c="k")
                ax.legend(loc="upper left", frameon=False)
                ax.set_ylim(
                    args.y_vals_limit[0] - 0.05, args.y_vals_limit[1] + 0.05
                )
                # ax.set(
                #     xlabel="Lag (s)",
                #     ylabel="Correlation (r)",
                #     title=f"{sid} {electrode} {row[3]}",
                # )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )
        pdf.savefig(fig)
        plt.close()
    return pdf


def main():
    args = arg_parser()
    args = set_up_environ(args)  # additional indirect args

    df = aggregate_data(args)
    df = organize_data(args, df)
    breakpoint()

    pdf = PdfPages(args.outfile)
    if len(args.y_vals_limit) == 1:  # automatic y limit
        args.y_vals_limit = [df.min().min(), df.max().max()]
    pdf = plot_average_split_by_key(args, df, pdf)
    pdf = plot_electrodes_split_by_key(args, df, pdf)

    pdf.close()

    return None


if __name__ == "__main__":
    main()
