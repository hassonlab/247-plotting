import glob
import argparse
import os
from re import L
import pandas as pd
import itertools
import numpy as np
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tfsplt_utils import read_sig_file, read_folder, read_folder2
from utils import main_timer


# -----------------------------------------------------------------------------
# Argument Parser Functions
# -----------------------------------------------------------------------------


def arg_parser():  # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--sid", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument("--fig-size", nargs="+", type=int, default=[18, 6])
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--x-vals-show", nargs="+", type=float, required=True)
    parser.add_argument("--lag-ticks", nargs="+", type=float, default=[])
    parser.add_argument("--lag-tick-labels", nargs="+", type=int, default=[])
    parser.add_argument("--lc-by", type=str, default=None)
    parser.add_argument("--ls-by", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--split-by", type=str, default=None)
    parser.add_argument("--outfile", default="results/figures/tmp.pdf")
    args = parser.parse_args()
    return args


def set_up_environ(args):

    args.x_vals_show = [x_val / 1000 for x_val in args.x_vals_show]
    args.lags_plot = [lag / 1000 for lag in args.lags_plot]
    args.lags_show = [lag / 1000 for lag in args.lags_show]
    args.unique_labels = list(dict.fromkeys(args.labels))
    args.unique_keys = list(dict.fromkeys(args.keys))

    args = get_cmap_smap(args)  # get color and style map

    return args


def arg_assert(args):  # some sanity checks
    assert len(args.fig_size) == 2
    assert len(args.formats) == len(
        args.labels
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

    if args.split:
        assert args.split_by, "Need split by criteria"
        assert args.split == "horizontal" or args.split == "vertical"
        assert args.split_by == "keys" or args.split_by == "labels"

    return args


# -----------------------------------------------------------------------------
# Utils Functions
# -----------------------------------------------------------------------------


def get_sigelecs(args):
    """Get significant electrodes

    Args:
        args (namespace): commandline arguments

    Returns:
        sigelecs (Dict): mode (key): list of sig elecs (value)
    """
    sigelecs = {}
    if len(args.sig_elec_file) == 0:
        pass
    elif len(args.sig_elec_file) == len(args.sid) * len(args.keys):
        sid_key_tup = [x for x in itertools.product(args.sid, args.keys)]
        for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
            sigelecs[sid_key] = read_sig_file(fname)
    else:
        raise Exception(
            "Need a significant electrode file for each subject-key combo"
        )
    return sigelecs


def add_sid(df, elec_name_dict):
    elec_name = df.index.to_series().str.get(1).tolist()
    sid_name = df.index.to_series().str.get(3).tolist()
    for idx, string in enumerate(elec_name):
        if string.find("_") < 0 or not string[0:3].isdigit():  # no sid in front
            new_string = str(sid_name[idx]) + "_" + string  # add sid
            elec_name_dict[string] = new_string
    return elec_name_dict


def sep_sid_elec(string):
    """Separate string into subject id and electrode name

    Args:
        string: string in the format

    Returns:
        tuple in the format (subject id, electrode name)
    """
    sid_index = string.find("_")
    if sid_index > 1:  # if string contains '_'
        if string[:sid_index].isdigit():  # if electrode name starts with sid
            sid_name = string[:sid_index]
            elec_name = string[(sid_index + 1) :]  # remove the sid
    return (sid_name, elec_name)


# -----------------------------------------------------------------------------
# Functions for Color and Style Maps
# -----------------------------------------------------------------------------


def colorFader(c1, c2, mix):
    """Get color in between two colors (based on linear interpolate)

    Args:
        c1: color 1 in hex format
        c2: color 2 in hex format
        mix: percentage between two colors (0 is c1, 1 is c2)

    Returns:
        a color in hex format
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_cmap_smap(args):
    """Get line color and style map for given label key combinations

    Args:
        args (namespace): commandline arguments

    Returns:
        cmap: dictionary of {line color: (label, key)}
        smap: dictionary of {line style: (label, key)}
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]  # separate colors

    cmap = plt.cm.get_cmap("jet")

    if len(args.labels) > 10:
        col_len = 17
        colors = [cmap(i / col_len) for i in range(1, col_len)]

    # colors = [colorFader('#97baf7','#000308',i/col_len) for i in range(1,col_len)] # color gradient

    # colors2 = [colorFader('#97baf7','#032661',i/col_len) for i in range(1,col_len)] # color gradient
    # colors1 = [colorFader('#f2b5b1','#6e0801',i/col_len) for i in range(1,col_len)]
    # colors = colors1 + colors2
    styles = ["-", "--", "-.", ":"]
    cmap = {}  # line color map
    smap = {}  # line style map

    if (
        args.lc_by == "labels" and args.ls_by == "keys"
    ):  # line color by labels and line style by keys
        for label, color in zip(args.unique_labels, colors):
            for key, style in zip(args.unique_keys, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == "keys" and args.ls_by == "labels"
    ):  # line color by keys and line style by labels
        for key, color in zip(args.unique_keys, colors):
            for label, style in zip(args.unique_labels, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "labels"
    ):  # both line color and style by labels
        for label, color, style in zip(args.unique_labels, colors, styles):
            for key in args.unique_keys:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "keys"
    ):  # both line color and style by keys
        for key, color, style in zip(args.unique_keys, colors, styles):
            for label in args.unique_labels:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    else:
        raise Exception("Invalid input for arguments lc_by or ls_by")

    args.cmap = cmap
    args.smap = smap
    return args


# -----------------------------------------------------------------------------
# Aggregate Data Functions
# -----------------------------------------------------------------------------


def aggregate_data(args, sigelecs, parallel=True):
    data = []
    print("Aggregating data")
    for fmt, label in zip(args.formats, args.labels):
        load_sid = 0
        for sid in args.sid:
            if str(sid) in fmt:
                load_sid = sid
        assert (
            load_sid != 0
        ), f"Need subject id for format {fmt}"  # check subject id for format is provided
        for key in args.keys:
            fname = fmt % key
            if "/matlab-" in fmt:
                data = read_folder2(data, fname, load_sid, label, key, "all")
            else:
                data = read_folder(
                    data,
                    fname,
                    sigelecs,
                    (load_sid, key),
                    load_sid,
                    label,
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

    df.set_index(["label", "electrode", "mode", "sid"], inplace=True)
    df = df.drop(columns="type")

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


# -----------------------------------------------------------------------------
# Plotting Average and Individual Electrodes
# -----------------------------------------------------------------------------


def get_elecbrain(electrode):
    """Get filepath for small brain plots

    Args:
        electrode: electrode name

    Returns:
        imname: filepath for small brain plot for the given electrode
    """
    sid, electrode = sep_sid_elec(electrode)
    if sid == "7170":
        sid = "717"
    elecdir = f"/projects/HASSON/247/data/elecimg/{sid}/"
    name = electrode.replace("EEG", "").replace("REF", "").replace("\\", "")
    name = name.replace("_", "").replace("GR", "G")
    imname = elecdir + f"thumb_{name}.png"  # + f'{args.sid}_{name}.png'
    return imname


def plot_average(args, df, pdf):
    print("Plotting Average")
    fig, ax = plt.subplots(figsize=(args.fig_size[0], args.fig_size[1]))
    # axins = inset_axes(ax,width=3,height=1.5,loc=2,borderpad=4)
    for mode, subdf in df.groupby(["label", "mode"], axis=0):
        vals = subdf.mean(axis=0)
        err = subdf.sem(axis=0)
        label = "-".join(mode)
        ax.fill_between(
            args.x_vals_show,
            vals - err,
            vals + err,
            alpha=0.2,
            color=args.cmap[mode],
        )
        ax.plot(
            args.x_vals_show,
            vals,
            label=f"{label} ({len(subdf)})",
            color=args.cmap[mode],
            ls=args.smap[mode],
        )
        # layer_num = int(mode[0].replace('layer',''))
        # axins.scatter(layer_num, max(vals), color=args.cmap[mode])
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0 - 0.005, 0.3)  # .35
    ax.set(xlabel="Lag (s)", ylabel="Correlation (r)", title="Global average")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_split_by_key(args, df, pdf):
    if args.split == "horizontal":
        print("Plotting Average split horizontally by keys")
        fig, axes = plt.subplots(
            1,
            len(args.unique_keys),
            figsize=(args.fig_size[0], args.fig_size[1]),
        )
    else:
        print("Plotting Average split vertically by keys")
        fig, axes = plt.subplots(
            len(args.unique_keys),
            1,
            figsize=(args.fig_size[0], args.fig_size[1]),
        )
    for ax, (mode, subdf) in zip(axes, df.groupby("mode", axis=0)):
        for label, subsubdf in subdf.groupby("label", axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            key = (label, mode)
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
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(mode + " global average")
        ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_split_by_label(args, df, pdf):
    if args.split == "horizontal":
        print("Plotting Average split horizontally by labels")
        fig, axes = plt.subplots(
            1,
            len(args.unique_labels),
            figsize=(args.fig_size[0], args.fig_size[1]),
        )
    else:
        print("Plotting Average split vertically by labels")
        fig, axes = plt.subplots(
            len(args.unique_labels),
            1,
            figsize=(args.fig_size[0], args.fig_size[1]),
        )
    for ax, (label, subdf) in zip(axes, df.groupby("label", axis=0)):
        for mode, subsubdf in subdf.groupby("mode", axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            key = (label, mode)
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
                label=f"{mode} ({len(subsubdf)})",
                color=args.cmap[key],
                ls=args.smap[key],
            )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(label + " global average")
        ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes(args, df, pdf, vmin, vmax):
    print("Plotting Individual Electrodes")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, ax = plt.subplots(figsize=(args.fig_size[0], args.fig_size[1]))
        # axins = inset_axes(ax,width=3,height=1.5,borderpad=4)
        for (label, _, mode, _), values in subdf.iterrows():
            mode = (label, mode)
            label = "-".join(mode)
            ax.plot(
                args.x_vals_show,
                values,
                label=label,
                color=args.cmap[mode],
                ls=args.smap[mode],
            )
            # layer_num = int(mode[0].replace('layer',''))
            # axins.scatter(layer_num, max(values), color=args.cmap[mode])
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
        ax.legend(loc="upper left", frameon=False)
        ax.set(
            xlabel="Lag (s)",
            ylabel="Correlation (r)",
            title=f"{sid} {electrode}",
        )
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


def plot_electrodes_split_by_key(args, df, pdf, vmin, vmax):
    print("Plotting Individual Electrodes split by keys")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        if args.split == "horizontal":
            fig, axes = plt.subplots(
                1,
                len(args.unique_keys),
                figsize=(args.fig_size[0], args.fig_size[1]),
            )
        else:
            fig, axes = plt.subplots(
                len(args.unique_keys),
                1,
                figsize=(args.fig_size[0], args.fig_size[1]),
            )
        for ax, (mode, subsubdf) in zip(axes, subdf.groupby("mode")):
            for row, values in subsubdf.iterrows():
                label = row[0]
                key = (label, mode)
                ax.plot(
                    args.x_vals_show,
                    values,
                    label=label,
                    color=args.cmap[key],
                    ls=args.smap[key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper left", frameon=False)
            ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {mode}",
            )
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


def plot_electrodes_split_by_label(args, df, pdf, vmin, vmax):
    print("Plotting Individual Electrodes split by labels")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        if args.split == "horizontal":
            fig, axes = plt.subplots(
                1,
                len(args.unique_labels),
                figsize=(args.fig_size[0], args.fig_size[1]),
            )
        else:
            fig, axes = plt.subplots(
                len(args.unique_labels),
                1,
                figsize=(args.fig_size[0], args.fig_size[1]),
            )
        for ax, (label, subsubdf) in zip(axes, subdf.groupby("label")):
            for row, values in subsubdf.iterrows():
                mode = row[2]
                key = (label, mode)
                ax.plot(
                    args.x_vals_show,
                    values,
                    label=mode,
                    color=args.cmap[key],
                    ls=args.smap[key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper left", frameon=False)
            ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {label}",
            )
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


@main_timer
def main():

    args = arg_parser()
    args = set_up_environ(args)  # additional indirect args
    arg_assert(args)  # some sanity checks

    sigelecs = get_sigelecs(args)
    df = aggregate_data(args, sigelecs)
    df = organize_data(args, df)

    pdf = PdfPages(args.outfile)
    vmax, vmin = df.max().max(), df.min().min()
    if args.split:
        if args.split_by == "keys":
            pdf = plot_average_split_by_key(args, df, pdf)
            pdf = plot_electrodes_split_by_key(args, df, pdf, vmin, vmax)
        elif args.split_by == "labels":
            pdf = plot_average_split_by_label(args, df, pdf)
            pdf = plot_electrodes_split_by_label(args, df, pdf, vmin, vmax)
    else:
        pdf = plot_average(args, df, pdf)
        pdf = plot_electrodes(args, df, pdf, vmin, vmax)

    pdf.close()

    return None


if __name__ == "__main__":
    main()
