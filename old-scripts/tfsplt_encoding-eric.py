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
from tfsplt_utils import read_data, read_sig_file


def get_child_folders(args):
    folders = {}
    for condition in args.conditions:
        format = glob.glob(
            args.top_dir + f"/*-{condition}-*"
        )  # all children with the formats
        if condition == "all":
            format = [
                ind_format for ind_format in format if "flip" not in ind_format
            ]  # blenderbot
        # format = [ind_format for ind_format in format if '-1' in ind_format] # for testing
        folders[condition] = format
    return folders


def get_sigelecs(args):
    # Sig Elecs
    sigelecs = {}
    if args.sig_elecs:
        for mode in args.modes:
            sig_file_name = f"tfs-sig-file-{str(args.sid)}-sig-1.0-{mode}.csv"
            sigelecs[mode] = read_sig_file(sig_file_name)
    return sigelecs


def parse_arguments():
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


def set_up_environ(args):
    args.layers = np.arange(1, args.layer_num + 1)
    args.sigelecs = get_sigelecs(args)
    args.lags = np.arange(-2000, 2001, 25)

    return args


def get_layer_ctx(args, dir_path):
    # get layer number from directory path
    layer_idx = dir_path.rfind("-")
    layer = dir_path[(layer_idx + 1) :]
    assert layer.isdigit(), "Need layer to be an int"
    layer = int(layer)

    # get context length from directory path
    if args.has_ctx:
        partial_format = dir_path[:layer_idx]
        ctx_len = partial_format[(partial_format.rfind("-") + 1) :]
        assert ctx_len.isdigit(), "Need context length to be an int"
        ctx_len = int(ctx_len)
    else:
        ctx_len = None
    return (layer, ctx_len)


def aggregate_data(dir_path, args, mode, condition):
    subdata = []
    files = glob.glob(dir_path + f"/*/*_{mode}.csv")
    subdata = read_folder2(subdata, files, args.sigelecs, mode)
    subdata = pd.concat(subdata)

    subdata_ave = subdata.describe().loc[["mean"],]
    layer, ctx_len = get_layer_ctx(args, dir_path)
    print("cond:", condition, " mode:", mode, " context:", ctx_len, " layer:", layer)
    subdata_ave = subdata_ave.assign(layer=layer, mode=mode, condition=condition)
    if args.has_ctx:
        subdata_ave = subdata_ave.assign(ctx_len=ctx_len)

    return subdata_ave


def aggregate_folders_parallel(args, folders):
    print("Aggregating Data")
    data = []
    p = Pool(10)
    for condition in args.conditions:
        for mode in args.modes:
            for result in p.map(
                partial(
                    aggregate_data,
                    args=args,
                    mode=mode,
                    condition=condition,
                ),
                folders[condition],
            ):
                data.append(result)

    print("Organizing Data")
    df = pd.concat(data)
    assert len(df) == args.layer_num * len(args.conditions) * len(args.modes)
    if args.has_ctx:
        df = df.sort_values(by=["condition", "mode", "ctx", "layer"])
        df.set_index(["condition", "mode", "ctx", "layer"], inplace=True)
    else:
        df = df.sort_values(by=["condition", "mode", "layer"])
        df.set_index(["condition", "mode", "layer"], inplace=True)

    return df


def ericplot1(args, df, pdf):
    cmap = plt.cm.get_cmap("jet")
    marker_diff = 0.01

    fig, axes = plt.subplots(
        len(args.conditions), len(args.modes), figsize=(18, 5 * len(args.conditions))
    )
    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i, j]
            encrs = df.loc[(condition, mode), :]
            max_lags = encrs.idxmax(axis=1)
            yheights = np.ones(encrs.shape[-1]) * 1.01 + marker_diff
            encrs = encrs.divide(encrs.max(axis=1), axis=0)
            for layer in args.layers:
                ax.plot(
                    args.lags,
                    encrs.loc[layer],
                    c=cmap(layer / args.layer_num),
                    zorder=-1,
                    lw=0.5,
                )
                maxlag = max_lags[layer]
                ax.scatter(
                    args.lags[maxlag],
                    yheights[maxlag],
                    marker="o",
                    color=cmap(layer / args.layer_num),
                )
                yheights[maxlag] += marker_diff
            ax.set(ylim=(0.8, 1.2), xlim=(-2000, 1000))
            if i == 0:
                ax.title.set_text(mode)
            if j == 0:
                ax.set_ylabel(condition)
            if j == 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.05)
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
                # cbar = fig.colorbar(sm, ax=ax)
                cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
                cbar.set_ticklabels(
                    [
                        1,
                        int(args.layer_num / 4),
                        int(args.layer_num / 2),
                        int(3 * args.layer_num / 4),
                        args.layer_num,
                    ]
                )

    pdf.savefig(fig)
    plt.close()
    return pdf


def ericplot2(args, df, pdf):
    colors = ("blue", "red", "green", "black")
    fig, axes = plt.subplots(
        len(args.conditions), len(args.modes), figsize=(18, 5 * len(args.conditions))
    )
    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i, j]
            encrs = df.loc[(condition, mode), :]
            max_time = args.lags[encrs.idxmax(axis=1)]  # find the lag for the maximum
            ax.scatter(args.layers, max_time, c=colors[j])
            # Fit line through scatter
            linefit = stats.linregress(args.layers, max_time)
            ax.plot(
                args.layers,
                args.layers * linefit.slope + linefit.intercept,
                ls="--",
                c=colors[j],
            )
            ax.set(ylim=(-2000, 1000))
            ax.text(
                3,
                ax.get_ylim()[1] - 200,
                f"r={linefit.rvalue:.2f} p={linefit.pvalue:.2f}",
            )
            if i == 0:
                ax.set_title(mode)
            if j == 0:
                ax.set_ylabel(condition)
    pdf.savefig(fig)
    plt.close()
    return pdf


def ericplot3(args, df, pdf, type):
    ticks = [1, args.layer_num / 2, args.layer_num]
    fig, axes = plt.subplots(
        len(args.conditions), len(args.modes), figsize=(18, 5 * len(args.conditions))
    )

    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i, j]

            if type == "mean(2s)":
                chosen_lag_idx = [
                    idx
                    for idx, element in enumerate(args.lags)
                    if element <= 2000 and element >= -2000
                ]
                encrs = df.loc[(condition, mode), chosen_lag_idx]
                mean = encrs.mean(axis=1)
                errs = encrs.std(axis=1) ** 2

                ax.bar(
                    args.layers,
                    mean,
                    yerr=errs,
                    align="center",
                    alpha=0.5,
                    ecolor="black",
                    capsize=0,
                )
            elif type == "max":
                encrs = df.loc[(condition, mode), :]
                max = encrs.max(axis=1)
                ax.scatter(args.layers, max)

            ax.set(xlim=(0.5, args.layer_num + 0.5), xticks=ticks, xticklabels=ticks)
            if i == 0:
                ax.set_title(f"{mode}-{type}")
            if j == 0:
                ax.set_ylabel(condition)

    pdf.savefig(fig)
    plt.close()
    return pdf


def main():
    args = parse_arguments()
    args = set_up_environ(args)

    folders = get_child_folders(args)  # get child folders
    df = aggregate_folders_parallel(args, folders)  # aggregate data

    print("Plotting")
    pdf = PdfPages(args.outfile)
    pdf = ericplot1(args, df, pdf)
    pdf = ericplot2(args, df, pdf)
    pdf = ericplot3(args, df, pdf, "mean(2s)")
    pdf = ericplot3(args, df, pdf, "max")
    pdf.close()

    return


if __name__ == "__main__":
    main()


def heatmap(args, df, pdf, type):
    # sns.dark_palette("#69d", reverse=True, as_cmap=True)
    plot_max = df.max(axis=1).unstack(0).sort_values(by="layer", ascending=False)
    plot_idxmax = df.idxmax(axis=1).unstack(0)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax = sns.heatmap(plot_max, cmap="Blues")
    pdf.savefig(fig)
    plt.close()
    return pdf


# plot_max = df.max(axis=1).unstack(0).sort_values(by='layer',ascending=False)
# plot_idxmax = df.idxmax(axis=1).unstack(0)
# idx_vals = list(range(-10000,10025,25))

# def get_idx_val(idx):
#     return idx_vals[idx]

# plot_idxmax = plot_idxmax.apply(np.vectorize(get_idx_val))
# sns.dark_palette("#69d", reverse=True, as_cmap=True)


# print('Plotting')
# pdf = PdfPages('results/figures/layer_ctx.pdf')

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.heatmap(plot_max, cmap='Blues')
# pdf.savefig(fig)
# plt.close()

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.heatmap(plot_idxmax, cmap='Blues')
# pdf.savefig(fig)
# plt.close()

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.scatterplot(x=plot_max.index, y=plot_max)
# pdf.savefig(fig)

# # plot_idxmax_melt = plot_idxmax.melt('layer',var_name='mode')
# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.scatterplot(x=plot_idxmax.index, y=plot_idxmax)
# pdf.savefig(fig)
# pdf.close()
