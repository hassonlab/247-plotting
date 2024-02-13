import argparse
import os
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tfsplt_brainmap import (
    get_sigelecs,
    Colorbar,
    read_coor,
    load_surf,
    plot_surf,
    update_properties,
    make_brainmap,
)
from tfsplt_encoding import organize_data
from tfsplt_utils import Colormap2D

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
    parser.add_argument("--sid", type=str, nargs="+", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--effect", nargs="?", type=str, required=True)
    parser.add_argument("--cmap", nargs="?", type=str, required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument(
        "--sig-elec-file-dir", nargs="?", default="data/plotting/sig-elecs/"
    )
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--final", action="store_true", default=False)
    parser.add_argument("--final2", action="store_true", default=False)
    parser.add_argument("--shiny", action="store_true", default=False)
    parser.add_argument("--outfile", default="results/figures/tfs-encoding.pdf")
    args = parser.parse_args()

    return args


def set_up_environ(args):
    """Adding necessary plotting information to args

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    args.sig_elec_file = [
        os.path.join(args.sig_elec_file_dir, file) for file in args.sig_elec_file
    ]

    # Additional args
    if "777" in args.sid:
        args.project = "podcast"
        assert len(args.sig_elec_file) <= 1
    else:
        args.project = "tfs"
    args.main_dir = "data/plotting/brainplot/"
    args.brain_type = "ave"  # ave or ind
    args.hemisphere = "both"  # left or right or both

    args = get_sigelecs(args)  # get significant electrodes

    return args


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


def aggregate_data(args):
    """Aggregate encoding data

    Args:
        args (namespace): commandline arguments

    Returns:
        df (DataFrame): df with all encoding results
    """
    print("Aggregating data")

    def read_file(fname):
        files = glob.glob(fname)
        assert (
            len(files) > 0
        ), f"No results found under {fname}"  # check files exist under format

        for resultfn in files:
            elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
            # elec = os.path.basename(resultfn).replace(".csv", "")[:-10]
            # Skip electrodes if they're not part of the sig list
            if len(args.sigelecs) and elec not in args.sigelecs[(load_sid, key)]:
                continue
            df = pd.read_csv(resultfn, header=None)
            # df = df.iloc[[11], :]
            df.insert(0, "sid", load_sid)
            df.insert(0, "key", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)

    data = []

    for load_sid in args.sid:
        for fmt, label in zip(args.formats, ["enca", "encb", "encab"]):
            for key in args.keys:
                fname = fmt % (load_sid, key)
                read_file(fname)

    if not len(data):
        print("No data found")
        exit(1)
    df = pd.concat(data)
    return df


def add_effect(args, df):
    """Adding effect column to dataframe

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results

    Returns:
        df (DataFrame): df with all encoding results and effect
        color_split (list): list of ints and Colorbar
    """

    def get_part_df(label):  # get partial df
        idx = pd.IndexSlice
        part_df = df.loc[idx[label, :, :, :], :].copy()
        part_df.index = part_df.index.droplevel("label")
        part_df_idx = part_df.index.get_level_values("electrode").tolist()
        return part_df, part_df_idx

    def rgb_to_hex(color):
        return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

    if len(args.formats) == 3:
        df["max"] = df.max(axis=1)
        df1, df1_idx = get_part_df("enca")
        df2, df2_idx = get_part_df("encb")
        df3, df3_idx = get_part_df("encab")
        assert len(df1_idx) == len(df2_idx) == len(df3_idx)
        assert all([a == b for a, b in zip(df1_idx, df2_idx)])
        assert all([a == b for a, b in zip(df1_idx, df3_idx)])

        df1.loc[:, "shared_var"] = df1["max"] ** 2 + df2["max"] ** 2 - df3["max"] ** 2
        df1.loc[df1.shared_var < 0, "shared_var"] = 0
        df1.loc[:, "shared"] = np.sqrt(df1.shared_var)
        df1.loc[:, "ua_var"] = df1["max"] ** 2 - df1["shared"] ** 2
        df1.loc[df1.ua_var < 0, "ua_var"] = 0
        df1.loc[:, "ua"] = np.sqrt(df1.ua_var)
        df1.loc[:, "ub_var"] = df2["max"] ** 2 - df1["shared"] ** 2
        df1.loc[df1.ub_var < 0, "ub_var"] = 0
        df1.loc[:, "ub"] = np.sqrt(df1.ub_var)
        df1.loc[:, "effect1"] = df1.ua**2 / df3["max"] ** 2
        df1.loc[:, "effect2"] = df1.ub**2 / df3["max"] ** 2
        # df1.loc[:, "effect1"] = df1.ua
        # df1.loc[:, "effect2"] = df1.ub
        # df1.loc[:, "effect1"] = df1["max"] / df3["max"]
        # df1.loc[:, "effect2"] = df2["max"] / df3["max"]
        df = df1

        if args.effect == "varpar":
            cc = Colormap2D(
                args.cmap,
                vmin=0,
                vmax=1,
                vmin2=0,
                vmax2=1,
                vflip=True,
                hflip=True,
            )
            red, green, blue, alpha = cc(df.loc[:, ("effect1", "effect2")].to_numpy())
            colors = np.vstack((red, green, blue, alpha)).T
            colors_hex = [rgb_to_hex(color) for color in colors]
            df["effect"] = colors_hex
            df.reset_index(inplace=True)
        elif args.effect == "shared":
            df["effect"] = df1.shared / df3["max"]
            df.reset_index(inplace=True)
            args.color_split = [Colorbar(bar_min=0, bar_max=1)]

    elif len(args.formats) == 2:
        df["max"] = df.max(axis=1)
        df1, df1_idx = get_part_df("enca")
        df2, df2_idx = get_part_df("encb")
        assert len(df1_idx) == len(df2_idx)
        assert all([a == b for a, b in zip(df1_idx, df2_idx)])
        df1.loc[:, "max2"] = df2["max"]
        df = df1
        # df["max3"] = df.loc[:, ("max", "max2")].max(axis=1) # ratio
        # df["max"] = df["max"] / df["max3"]
        # df["max2"] = df["max2"] / df["max3"]
        # cc = Colormap2D(args.cmap, vmin=0, vmax=1, vmin2=0, vmax2=1)
        cc = Colormap2D(args.cmap, vmin=0, vmax=0.3, vmin2=0, vmax2=0.3)
        red, green, blue, alpha = cc(df.loc[:, ("max2", "max")].to_numpy())
        colors = np.vstack((red, green, blue, alpha)).T
        colors_hex = [rgb_to_hex(color) for color in colors]
        df["effect"] = colors_hex
        df.reset_index(inplace=True)

    return df


# -----------------------------------------------------------------------------
# Brain Map
# -----------------------------------------------------------------------------


def plot_electrodes(fig, df, args):
    """Plot electrodes onto figure

    Args:
        fig (plotly graph object): brain map plot
        df (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object with title and colorscale

    Returns:
        fig (plotly graph object): brain map plot with electrodes added
    """
    print(f"Plot {len(df)} Electrodes with {args.brain_type} coordinates")
    if args.brain_type == "ave":
        coor_type = "MNI"
    elif args.brain_type == "ind":
        coor_type = "T1"
        assert len(df.subject.unique()) == 1  # only 1 subject
    r = 1.5
    legend_show = True
    if args.final or args.final2:
        legend_show = False
    if args.final2:
        r = 1.8
    for center_x, center_y, center_z, effect in zip(
        df[f"{coor_type}_X"],
        df[f"{coor_type}_Y"],
        df[f"{coor_type}_Z"],
        df.effect,
    ):
        u, v = np.mgrid[0 : 2 * np.pi : 26j, 0 : np.pi : 26j]
        x = r * np.cos(u) * np.sin(v) + center_x
        y = r * np.sin(u) * np.sin(v) + center_y
        z = r * np.cos(v) + center_z
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=np.full(shape=z.shape, fill_value=effect),
                name="Elec",
                legendgroup="Elec",
                colorscale=[[0, effect], [1, effect]],
                showlegend=legend_show,
                showscale=False,
            )
        )
        legend_show = False
    if args.shiny:  # add light to elecs
        fig.update_traces(lighting_specular=0.4)

    return fig


def plot_brainmap_cat(args, df, outfile):
    """Plot brainmap plot given a df file with coordinates and effects

    Args:
        df (pandas DataFrame): DataFrame with electrode coordinates and effects
        outfile (str): outfile name

    Returns:
        fig (plotly graph object): brain map plot
    """
    # Plot Brain Surface
    surf1, surf2 = load_surf(args)
    fig = go.Figure()
    if surf1:  # left hemisphere
        fig = plot_surf(fig, surf1, args)
    if surf2:  # right hemisphere
        fig = plot_surf(fig, surf2, args)

    # Plot Electrodes and Colorbars
    fig = plot_electrodes(fig, df, args)

    # Save Plot
    fig = update_properties(fig, args)
    fig.update_layout(legend=dict(font=dict(size=30)))
    width = 1200
    if args.final or args.final2:
        width = 1000
    try:
        print(f"Writing to {outfile}")
        fig.write_image(outfile, scale=36, width=width, height=1000)
    except:
        print("Not writing to file")
    return fig


def make_brainmap_cat(args, df, outfile=""):
    """Plot and Save brainmap plot given a pandas Series of effects

    Args:
        args (namespace): commandline arguments
        df (pandas DataFrame): df with "electrode" and "effect" columns
        outfile (str): outfile name

    Returns:
        fig (plotly graph object): brain map plot
    """
    if len(df) == 0:
        print("Empty Dataframe")
        return
    print(f"Number of electrodes for encoding: {len(df)}")
    if args.project == "tfs":
        df = df.assign(
            subject=df.electrode.str.split("_", n=1, expand=True)[0],
            electrode=df.electrode.str.split("_", n=1, expand=True)[1],
        )
        df.loc[df.subject == "7170", "subject"] = 717  # fix for 717

        # Get Electrode Coordinate Files
        subjects = df.subject.unique()
        df_coor = read_coor(args.main_dir, subjects)
        df_plot = pd.merge(
            df.loc[:, ("subject", "electrode", "effect")],
            df_coor,
            how="inner",
            left_on=["subject", "electrode"],
            right_on=["subject", "name"],
        )
    elif args.project == "podcast":
        df_coor = pd.read_csv(  # Get Electrode Coordinate File
            os.path.join(args.main_dir, "777/777_ave.txt"), sep=" ", header=None
        )
        df_coor.columns = ["electrode", "MNI_X", "MNI_Y", "MNI_Z", "Area"]
        df_plot = pd.merge(  # merge two files
            df.loc[:, ("electrode", "effect")],
            df_coor,
            how="inner",
            on="electrode",
        )
    print(f"Number of electrodes for plotting: {len(df_plot)}")

    # Plot Brainmap
    fig = plot_brainmap_cat(args, df_plot, outfile=outfile)
    return fig


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Get effect
    df = aggregate_data(args)
    df = organize_data(args, df)
    df = add_effect(args, df)

    for key in args.keys:
        try:
            outfile = args.outfile % key
        except:
            outfile = args.outfile

        if args.effect == "shared":
            make_brainmap(
                args,
                df.loc[df.key == key, ("electrode", "effect")],
                outfile,
            )
        else:
            make_brainmap_cat(
                args,
                df.loc[df.key == key, ("electrode", "effect")],
                outfile,
            )
    return


if __name__ == "__main__":
    main()
