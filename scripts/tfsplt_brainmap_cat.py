import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tfsplt_brainmap import (
    Colorbar,
    read_coor,
    load_surf,
    plot_surf,
    update_properties,
)

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
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument(
        "--sig-elec-file-dir", nargs="?", default="data/plotting/sig-elecs/"
    )
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
        assert len(args.sig_elec_file) == 1
    else:
        args.project = "tfs"
    args.main_dir = "data/plotting/brainplot/"
    args.brain_type = "ave"  # ave or ind
    args.hemisphere = "left"  # left or right or both

    return args


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


def get_sigelec_df(args):
    """Loading significant electrode list to dataframe for plotting

    Args:
        args (namespace): commandline arguments

    Returns:
        df (DataFrame): DataFrame with "subject" and "electrode" columns
    """
    if args.project == "podcast":  # podcast
        df = pd.read_csv(args.sig_elec_file[0])
        df["key"] = "comp"
    elif len(args.sig_elec_file) == 1:
        args.sig_elec_file = args.sig_elec_file[0]
        df = pd.DataFrame()
        for sid in args.sid:
            for key in args.keys:
                try:
                    filename = args.sig_elec_file % (sid, key)
                except:
                    dct = {"sid": sid, "key": key}
                    filename = args.sig_elec_file % dct
                sid_key_df = pd.read_csv(filename)
                sid_key_df["key"] = key
                df = pd.concat((df, sid_key_df))
    elif len(args.sig_elec_file) == 2:  # separate comp/prod
        df = pd.DataFrame()
        for sid in args.sid:
            for fname, key in zip(args.sig_elec_file, args.keys):
                filename = fname % sid
                sid_key_df = pd.read_csv(filename)
                sid_key_df["key"] = key
                df = pd.concat((df, sid_key_df))
    return df


# -----------------------------------------------------------------------------
# Brain Map
# -----------------------------------------------------------------------------


def plot_electrodes(fig, df, color, brain_type):
    """Plot electrodes onto figure

    Args:
        fig (plotly graph object): brain map plot
        df (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object with title and colorscale

    Returns:
        fig (plotly graph object): brain map plot with electrodes added
    """
    print(f"Plot {len(df)} Electrodes with {brain_type} coordinates")
    if brain_type == "ave":
        coor_type = "MNI"
    elif brain_type == "ind":
        coor_type = "T1"
        assert len(df.subject.unique()) == 1  # only 1 subject
    r = 1.5
    legend_show = True
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
                surfacecolor=np.full(shape=z.shape, fill_value=color),
                name=str(effect),
                legendgroup=str(effect),
                colorscale=[[0, color], [1, color]],
                showlegend=legend_show,
                showscale=False,
            )
        )
        legend_show = False

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
        fig = plot_surf(fig, surf1)
    if surf2:  # right hemisphere
        fig = plot_surf(fig, surf2)

    # Plot Electrodes and Colorbars
    for (_, df_colorscale), color in zip(df.groupby(df.effect), args.colors):
        fig = plot_electrodes(fig, df_colorscale, color, args.brain_type)

    # Save Plot
    fig = update_properties(fig)
    fig.update_layout(legend=dict(font=dict(size=30)))
    try:
        print(f"Writing to {outfile}")
        fig.write_image(outfile, scale=6, width=1200, height=1000)
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
    df = get_sigelec_df(args)

    df["effect"] = df.subject.astype(int)
    df.loc[df.effect == 7170, "effect"] = 717  # fix for 717
    color_list = px.colors.qualitative.D3  # 10 colors
    args.colors = color_list

    for key in args.keys:
        try:
            outfile = args.outfile % key
        except:
            outfile = args.outfile
        make_brainmap_cat(
            args,
            df.loc[df.key == key, ("subject", "electrode", "effect")],
            outfile,
        )
    return


if __name__ == "__main__":
    main()
