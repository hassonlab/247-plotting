import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from nilearn.plotting import plot_markers

from tfsplt_brainmap import (
    get_sigelecs,
    Colorbar,
    read_coor,
    load_surf,
    plot_surf,
    update_properties,
)
from tfsplt_brainmap_2d import (
    arg_parser,
    set_up_environ,
    aggregate_data,
)
from tfsplt_encoding import organize_data
from tfsplt_utils import Colormap2D


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


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

    if len(args.formats) == 4:
        df["max"] = df.max(axis=1)
        df1, df1_idx = get_part_df("enca")
        df2, df2_idx = get_part_df("encb")
        df3, df3_idx = get_part_df("encc")
        df4, df4_idx = get_part_df("encd")
        assert len(df1_idx) == len(df2_idx) == len(df3_idx) == len(df4_idx)
        assert all([a == b for a, b in zip(df1_idx, df2_idx)])
        assert all([a == b for a, b in zip(df1_idx, df3_idx)])
        assert all([a == b for a, b in zip(df1_idx, df4_idx)])

        df1.loc[:, "max2"] = df2["max"]
        df1.loc[:, "max3"] = df3["max"]
        df1.loc[:, "max4"] = df4["max"]
        df1.loc[:, "effect"] = df1["max"] + df1["max2"] - df1["max3"] - df1["max4"]
        df = df1
        df.reset_index(inplace=True)

    elif len(args.formats) == 3:
        df["max"] = df.max(axis=1)
        df1, df1_idx = get_part_df("enca")
        df2, df2_idx = get_part_df("encb")
        df3, df3_idx = get_part_df("encc")
        assert len(df1_idx) == len(df2_idx) == len(df3_idx)
        assert all([a == b for a, b in zip(df1_idx, df2_idx)])
        assert all([a == b for a, b in zip(df1_idx, df3_idx)])

        df1.loc[:, "max1"] = df1["max"]
        df1.loc[:, "max2"] = df2["max"]
        df1.loc[:, "max3"] = df3["max"]

        if args.effect == "color":
            print("3 color effect")
            df1.loc[:, "effect"] = 1
            df1.loc[df1.max2.ge(df1.max1) & df1.max2.ge(df1.max3), "effect"] = 2
            df1.loc[df1.max3.ge(df1.max1) & df1.max3.ge(df1.max2), "effect"] = 3

        elif args.effect == "gradient":
            print("Color gradient effect")
            df1.loc[:, "efcol"] = 2
            df1.loc[:, "efnum"] = 4
            df1.loc[df1.max2.ge(df1.max1) & df1.max2.ge(df1.max3), "efcol"] = 0
            df1.loc[df1.max3.ge(df1.max1) & df1.max3.ge(df1.max2), "efcol"] = 1
            df1 = df1.loc[df1[["max1", "max2", "max3"]].max(axis=1).ge(0), :]
            df1.loc[df1[["max1", "max2", "max3"]].max(axis=1).ge(0.1), "efnum"] = 3
            df1.loc[df1[["max1", "max2", "max3"]].max(axis=1).ge(0.2), "efnum"] = 2
            df1.loc[df1[["max1", "max2", "max3"]].max(axis=1).ge(0.3), "efnum"] = 1
            df1.loc[:, "effect"] = df1.efcol * 4 + df1.efnum

        elif args.effect == "varpar":
            print("Variance Partitioning")
            df1.loc[:, "shared_var"] = (
                df1["max"] ** 2 + df2["max"] ** 2 - df3["max"] ** 2
            )
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
            df1.loc[:, "effect"] = df1.effect1 - df1.effect2

        df = df1
        df.reset_index(inplace=True)

    elif len(args.formats) == 2:
        if args.effect == "gradient":
            df["max"] = df.max(axis=1)
            df1, df1_idx = get_part_df("enca")
            df2, df2_idx = get_part_df("encb")
            assert len(df1_idx) == len(df2_idx)
            assert all([a == b for a, b in zip(df1_idx, df2_idx)])
            df1.loc[:, "max2"] = df2["max"]
            df1.loc[:, "effect"] = df1["max2"] - df1["max"]

        elif args.effect == "color":
            df1 = df1.loc[df1.effect != 0, :]
            df1.loc[df1.effect > 0, "effect"] = 0.7
            df1.loc[df1.effect < 0, "effect"] = -0.7

        df = df1
        df.reset_index(inplace=True)

    elif len(args.formats) == 1:
        df["effect"] = df.max(axis=1)
        df.reset_index(inplace=True)
        print(df.effect.describe())

    return df


# -----------------------------------------------------------------------------
# Brain Map
# -----------------------------------------------------------------------------


def plot_glassbrain(args, df_plot, outfile=""):
    """Plot glass brain plot given a df file with coordinates and effects

    Args:
        df (pandas DataFrame): DataFrame with electrode coordinates and effects
        outfile (str): outfile name

    Returns:
        fig (matplotlib object): brain map plot
    """
    fig, axes = plt.subplots(1, 1, dpi=300, figsize=(8, 6))
    coords = np.array([df_plot.MNI_X, df_plot.MNI_Y, df_plot.MNI_Z]).T

    if args.effect == "color":
        colors = ["#d13843", "#669BBC", "#F9AD6A"]
        colors = ["#fcc317", "#6C8CBF", "#d14952"]
        cmap = mcolors.LinearSegmentedColormap.from_list("3col", colors, N=256)
    else:
        cmap = args.cmap

    plot_markers(
        df_plot.effect,
        coords,
        node_size=20,
        display_mode="l",
        # node_vmin=0,
        # node_vmax=0.2,
        # node_vmin=-0.2,
        # node_vmax=0.2,
        node_vmin=1,
        node_vmax=20,
        figure=fig,
        axes=axes,
        alpha=0.8,
        node_cmap=cmap,
        colorbar=True,
    )
    plt.savefig(outfile)

    return


def make_glassbrain(args, df, outfile=""):
    """Plot and Save glass brain plot given a pandas Series of effects

    Args:
        args (namespace): commandline arguments
        df (pandas DataFrame): df with "electrode" and "effect" columns
        outfile (str): outfile name

    Returns:
        fig (matplotlib object): brain map plot
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
    fig = plot_glassbrain(args, df_plot, outfile=outfile)
    return fig


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Get effect
    df = aggregate_data(args)
    df = organize_data(args, df)
    df = add_effect(args, df)

    # blue = "#0000ff"
    # red = "#ff0000"
    # df = df.loc[df.effect1 != df.effect2, :]
    # df.loc[df.effect1 > df.effect2, "effect"] = red
    # df.loc[df.effect1 < df.effect2, "effect"] = blue

    for key in args.keys:
        try:
            outfile = args.outfile % key
        except:
            outfile = args.outfile

        make_glassbrain(
            args,
            df.loc[df.key == key, ("electrode", "effect")],
            outfile,
        )
    return


if __name__ == "__main__":
    main()
