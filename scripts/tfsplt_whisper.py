import argparse
import glob
import itertools
import os
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr

#####################
## ARGUMENT PARSER ##
#####################

def arg_parser():  # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--sid", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--roi", type=str)
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
        lag in x_vals_show for lag in args.lag_ticks
    ), "X values show should contain all values from lags ticks"
    assert all(
        lag in lags_show for lag in args.lag_tick_labels
    ), "Lags show should contain all values from lag tick labels"
    assert len(args.lag_ticks) == len(
        args.lag_tick_labels
    ), "Need same number of lag ticks and lag tick labels"

    if args.split:
        assert args.split_by, "Need split by criteria"
        assert args.split == "horizontal" or args.split == "vertical"
        assert args.split_by == "keys" or args.split_by == "labels"


args = arg_parser()
x_vals_show = [x_val / 1000 for x_val in args.x_vals_show]
lags_show = [lag / 1000 for lag in args.lags_show]
arg_assert(args)

##############################
## COLOR AND STYLE SETTINGS ##
##############################

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

    # set label and legend fontsize
    params = {'legend.fontsize': 25,
         'axes.labelsize': 50,
         'axes.titlesize': 65,
         'xtick.labelsize': 50,
         'ytick.labelsize': 50}
    mpl.rcParams.update(params)


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

    # col_len1 = 1 
    # colors1 = [colorFader('#00008B','#0000FF',i/col_len1) for i in range(1,col_len1+1)] # color gradient - BLUE for decoder
    # col_len3 = 1
    # colors3 = [colorFader('#FFA500','#8C000F',i/col_len3) for i in range(1,col_len3+1)] # color gradient - RED for encoder

    #HACK
    col_len1 = 1 
    colors1 = [colorFader('#FFA500','#8C000F',i/col_len1) for i in range(1,col_len1+1)] # color gradient - BLUE for decoder
    col_len3 = 1
    colors3 = [colorFader('#FFA500','#8C000F',i/col_len3) for i in range(1,col_len3+1)] # color gradient - RED for encoder


    colors = colors1 + colors3

    # col_len=1
    # colors1 = [colorFader('#929591','#929591',i/col_len) for i in range(1,col_len+1)]
    # colors2 = [colorFader('#15B01A','#15B01A',i/col_len) for i in range(1,col_len+1)]
    # colors3 = [colorFader('#0343DF','#0343DF',i/col_len) for i in range(1,col_len+1)]
    # col_len4 = 1
    # colors4 = [colorFader('#FFA500','#8C000F',i/col_len4) for i in range(1,col_len4+1)] # color gradient - RED for encoder

    # colors = colors1 + colors2 + colors3 + colors4

    # if args.keys == ["comp"]:
    #     styles = ["-"]
    # elif args.keys == ["prod"]:
    #     styles = ["-"]

    #HACK
    if args.keys == ["comp"]:
        styles = [":","-"]
    elif args.keys == ["prod"]:
        styles = [":","-"]

    cmap = {}  # line color map
    smap = {}  # line style map

    if (
        args.lc_by == "labels" and args.ls_by == "keys"
    ):  # line color by labels and line style by keys
        for label, color in zip(unique_labels, colors):
            for key, style in zip(unique_keys, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == "keys" and args.ls_by == "labels"
    ):  # line color by keys and line style by labels
        for key, color in zip(unique_keys, colors):
            for label, style in zip(unique_labels, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif args.lc_by == args.ls_by == "labels":  # both line color and style by labels
        for label, color, style in zip(unique_labels, colors, styles):
            for key in unique_keys:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif args.lc_by == args.ls_by == "keys":  # both line color and style by keys
        for key, color, style in zip(unique_keys, colors, styles):
            for label in unique_labels:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    else:
        raise Exception("Invalid input for arguments lc_by or ls_by")
    return (cmap, smap)


unique_labels = list(dict.fromkeys(args.labels))
unique_keys = list(dict.fromkeys(args.keys))
cmap, smap = get_cmap_smap(args)

######################################
## READ SIGNIFICANE ELECTRODE FILES ##
######################################

sigelecs = {}
multiple_sid = False  # only 1 subject
if len(args.sid) > 1:
    multiple_sid = True  # multiple subjects
if len(args.sig_elec_file) == 0:
    pass
elif len(args.sig_elec_file) == len(args.sid) * len(args.keys):
    sid_key_tup = [x for x in itertools.product(args.sid, args.keys)]
    for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
        #HACK
        sig_file = pd.read_csv(f"/scratch/gpfs/ln1144/247-plotting/data/plotting/ROI-tfs-sig-files/tfs-sig-files-{args.model}/" + fname)
        if sig_file.subject.nunique() == 1:
            sig_file["sid_electrode"] = (
                sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
            )
            elecs = sig_file["sid_electrode"].tolist()
            sigelecs[sid_key] = set(elecs)
        else:
            sig_file["sid_electrode"] = (
                sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
            )
            elecs = sig_file["sid_electrode"].tolist()
            sigelecs[sid_key] = set(elecs)
            multiple_sid = True
else:
    raise Exception("Need a significant electrode file for each subject-key combo")

####################
## AGGREGATE DATA ##
####################

print("Aggregating data")
data = []

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
        files = glob.glob(fname)
        assert (
            len(files) > 0
        ), f"No results found under {fname}"  # check files exist under format

        for resultfn in files:
            elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
            # Skip electrodes if they're not part of the sig list
            # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
            #     continue
            if len(sigelecs) and elec not in sigelecs[(load_sid, key)]:
                continue
            # READ CSV TO ANOTHER DF FOR EACH ELEC AND APPEND AVG AND SE COLUMN TO DF AS TO ROWS
            pre_df = pd.read_csv(resultfn)
            results = {'avg':pre_df.avg,'se':pre_df.se}
            df = pd.DataFrame(results)
            df = df.transpose()
            df = df.rename_axis('dtype').reset_index()
            df.insert(0, "sid", load_sid)
            df.insert(0, "mode", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)

if not len(data):
    print("No data found")
    exit(1)
df = pd.concat(data)
df.set_index(["label", "electrode", "mode", "sid", "dtype"], inplace=True)

n_lags, n_df = len(args.lags_plot), len(df.columns)
assert (
    n_lags == n_df
), "args.lags_plot length ({n_av}) must be the same size as results ({n_df})"


def add_sid(df, elec_name_dict):
    elec_name = df.index.to_series().str.get(1).tolist()
    sid_name = df.index.to_series().str.get(3).tolist()
    for idx, string in enumerate(elec_name):
        if string.find("_") < 0 or not string[0:3].isdigit():  # no sid in front
            new_string = str(sid_name[idx]) + "_" + string  # add sid
            elec_name_dict[string] = new_string
    return elec_name_dict


elec_name_dict = {}
# new_sid = df.index.to_series().str.get(1).apply(add_sid) # add sid if no sid in front
elec_name_dict = add_sid(df, elec_name_dict)
df = df.rename(index=elec_name_dict)  # rename electrodes to add sid in front

if len(args.lags_show) < len(
    args.lags_plot
):  # if we want to plot part of the lags and not all lags
    print("Trimming Data")
    chosen_lag_idx = [
        idx for idx, element in enumerate(args.lags_plot) if element in args.lags_show
    ]
    df = df.loc[:, chosen_lag_idx]  # chose from lags to show for the plot
    assert len(x_vals_show) == len(
        df.columns
    ), "args.lags_show length must be the same size as trimmed df column number"


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

########################
## PLOTTING FUNCTIONS ##
########################

def plot_average(pdf):
    print("Plotting Average")
    fig, ax = plt.subplots(figsize=fig_size)
    # axins = inset_axes(ax, width=3, height=1.5, loc=2, borderpad=4)
    for mode, subdf in df.groupby(["label", "mode"], axis=0):
        subdf = subdf.query("dtype == 'avg'")
        vals = subdf.mean(axis=0)
        err = subdf.sem(axis=0)
        ax.fill_between(x_vals_show, vals - err, vals + err, alpha=0.2, color=cmap[mode])
        ax.plot(
            x_vals_show,
            vals,
            label=f"n=({len(subdf)})",
            color=cmap[mode],
            ls=smap[mode],
            lw = 10
        )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    #ax.legend(loc="upper right", frameon=False)
    ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    ax.set_title(f"{args.roi} ({len(subdf)})", weight = "bold")
    ax.set_ylim(-0.05, 0.3)  # .35
    ax.set_yticks([0,0.1,0.2,0.3])
    pdf.savefig(fig)
    plt.close()
    return pdf

def plot_electrodes(pdf):
    print("Plotting Individual Electrodes")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, ax = plt.subplots(figsize=fig_size)
        for mode, subsubdf in subdf.groupby(["label", "mode"], axis=0):
            vals = subsubdf.loc[:,:,:,:,'avg'].squeeze(axis=0)
            err = subsubdf.loc[:,:,:,:,'se'].squeeze(axis=0)
            # vals = subsubdf.query("dtype == 'vals'").squeeze(axis=0)
            # err = subsubdf.query("dtype == 'se'").squeeze(axis=0)
            try:
                ax.fill_between(x_vals_show, vals - err, vals + err, alpha=0.2, color=cmap[mode])
            except:
                breakpoint()
            ax.plot(x_vals_show, vals, label=label, color=cmap[mode], ls=smap[mode], lw = 10)
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
        # ax.legend(loc="upper left", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)", title=f"{sid} {electrode}")
        ax.set_ylim(-0.05, 0.5)  # .35
        # ax.set_yticks([0,0.1,0.2,0.3])

        pdf.savefig(fig)
        plt.close()
    return pdf


##########
## MAIN ##
##########

pdf = PdfPages(args.outfile)

fig_size = (args.fig_size[0], args.fig_size[1])
vmax, vmin = df.max().max(), df.min().min()

pdf = plot_average(pdf)
pdf = plot_electrodes(pdf)

pdf.close()
