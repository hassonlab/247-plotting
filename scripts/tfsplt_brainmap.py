import argparse
import glob
import os
import io

import numpy as np
import pandas as pd
from scipy.io import loadmat
import plotly.graph_objects as go
from PIL import Image

from tfsplt_encoding import organize_data

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


def read_sig_file(filename):
    """Read significant electrode list
        sigelecs (Dict): Dictionary in the following format
            tuple of (sid,key) : [list of sig elec]

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    sig_file = pd.read_csv(filename)
    sig_file["sid_electrode"] = (
        sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
    )
    elecs = sig_file["sid_electrode"].tolist()

    return set(elecs)


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
    elif args.project == "podcast":  # podcast
        sigelecs[("777", "comp")] = read_sig_file(args.sig_elec_file[0])
    elif len(args.sig_elec_file) == 1:  # one sig file format
        args.sig_elec_file = args.sig_elec_file[0]
        for sid in args.sid:
            for key in args.keys:
                try:
                    filename = args.sig_elec_file % (sid, key)
                except:
                    dct = {"sid": sid, "key": key}
                    filename = args.sig_elec_file % dct
                sigelecs[(sid, key)] = read_sig_file(filename)
    elif len(args.sig_elec_file) == 2:  # separate comp/prod
        for sid in args.sid:
            for fname, key in zip(args.sig_elec_file, args.keys):
                filename = fname % sid
                sigelecs[(sid, key)] = read_sig_file(filename)

    args.sigelecs = sigelecs
    return args


def set_up_environ(args):
    """Adding necessary plotting information to args

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    args.lags_show = [lag / 1000 for lag in args.lags_show]
    args.lags_plot = [lag / 1000 for lag in args.lags_plot]
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
    if args.final or args.final2:
        args.hemisphere = "both"

    if "-diff" in args.effect:
        assert len(args.formats) == 2, "Provide 2 formats"
    else:
        assert len(args.formats) == 1, "Provide 1 format"

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
            # elec = os.path.basename(resultfn).replace(".csv", "")[:-10] # for fold
            # Skip electrodes if they're not part of the sig list
            if len(args.sigelecs) and elec not in args.sigelecs[(load_sid, key)]:
                continue
            df = pd.read_csv(resultfn, header=None)
            # df = df.T.iloc[[10], 1:]  # for Leo's results
            # df = df.iloc[[11], :] # for fold
            df.insert(0, "sid", load_sid)
            df.insert(0, "key", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)

    data = []

    for load_sid in args.sid:
        for fmt, label in zip(args.formats, ["enc1", "enc2"]):
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

    if args.effect == "max":
        df["effect"] = df.max(axis=1)
        color_split = [Colorbar(bar_min=0.04, bar_max=0.4)]
    elif args.effect == "mean":
        df["effect"] = df.mean(axis=1)
        color_split = [Colorbar()]
    elif args.effect == "max-diff":
        df["max"] = df.max(axis=1)
        df1, df1_idx = get_part_df("enc1")
        df2, df2_idx = get_part_df("enc2")
        assert len(df1_idx) == len(df2_idx)
        assert all([a == b for a, b in zip(df1_idx, df2_idx)])
        df1.loc[:, "effect"] = df1["max"] - df2["max"]
        df = df1
        color_split = [
            Colorbar(
                title="Δ corr neg",
                colorscale=[[0, "rgb(0,0,255)"], [1, "rgb(240,248,255)"]],
                bar_min=-0.15,
                bar_max=0,
            ),
            0,
            Colorbar(
                title="Δ corr pos",
                colorscale=[[0, "rgb(255,248,240)"], [1, "rgb(255,0,0)"]],
                bar_min=0,
                bar_max=0.15,
            ),
        ]
    elif args.effect == "area-diff":
        breakpoint()
        color_split = [
            Colorbar(
                title="area neg",
                colorscale=[[0, "rgb(0,0,255)"], [1, "rgb(240,248,255)"]],
            ),
            0,
            Colorbar(
                title="area pos",
                colorscale=[[0, "rgb(255,248,240)"], [1, "rgb(255,0,0)"]],
            ),
        ]

    df.reset_index(inplace=True)
    args.color_split = color_split

    return df


# -----------------------------------------------------------------------------
# Brain Map
# -----------------------------------------------------------------------------


def read_coor(path, subjects):
    """Read electrode coordinate files

    Args:
        path (str): electrode coordinate file directory
        subjects (list): list of subject id

    Returns:
        df_coor (DataFrame): pandas DataFrame of electrode coordinates
    """
    df_coor = pd.DataFrame()
    for sid in subjects:
        file = os.path.join(path, f"{sid}", f"{sid}-electrode-coordinates.csv")
        df = pd.read_csv(file)
        df.dropna(subset=["name"], inplace=True)
        df["subject"] = sid
        df_coor = pd.concat([df_coor, df])

    return df_coor


def load_surf(args):
    """Load brain surface mat files

    Args:
        path (str): brain surface mat file directory
        id (list): list of subject id
        hemisphere (str): hemisphere to load, either left, right, or both

    Returns:
        surf1 (dict): brain surface mat
        surf2 (dict): brain surface mat
    """

    if args.brain_type == "ave":
        file = glob.glob(os.path.join(args.main_dir, "*.mat"))
    else:
        assert len(args.sid) == 1  # only one subject
        file = glob.glob(
            os.path.join(args.main_dir, args.sid[0], f"NY{args.sid[0]}*.mat")
        )

    assert len(file) <= 2, "Duplicate surface mat files exists"
    if args.hemisphere == "left":  # left hemisphere
        surf1 = loadmat(file[0])
        surf2 = None
    elif args.hemisphere == "right":  # right hemisphere
        surf1 = None
        surf2 = loadmat(file[1])
    elif args.hemisphere == "both":  # both hemispheres
        surf1 = loadmat(file[0])
        surf2 = loadmat(file[1])
    else:
        surf1 = surf2 = None

    return surf1, surf2


def plot_surf(fig, surf, args):
    """Plot brain surface onto figure

    Args:
        fig (plotly graph object): brain map plot
        surf (dict): brain surface file

        Note: surf["faces"] is an n x 3 matrix of indices into surf["coords"]; connectivity matrix

    Returns:
        fig (plotly graph object): brain map plot with brain surface added
    """
    print("Plot Hemisphere")
    # Subtract 1 from every index to convert MATLAB indexing to Python indexing
    surf["faces"] = np.array([conn_idx - 1 for conn_idx in surf["faces"]])
    # Plot 3D surface plot of brain, colored according to depth
    fig.add_trace(
        go.Mesh3d(
            x=surf["coords"][:, 0],
            y=surf["coords"][:, 1],
            z=surf["coords"][:, 2],
            i=surf["faces"][:, 0],
            j=surf["faces"][:, 1],
            k=surf["faces"][:, 2],
            color="rgb(175,175,175)",
        )
    )

    if args.final:
        ambient = 0.22
        specular = 0.4
    elif args.final2:
        ambient = 0.3
        specular = 0.3
    else:
        ambient = 0.4
        specular = 0.4

    fig.update_traces(
        lighting_ambient=ambient,
        lighting_specular=specular,
    )
    return fig


def plot_electrodes(fig, df, cbar, args):
    """Plot electrodes onto figure

    Args:
        fig (plotly graph object): brain map plot
        df (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object with title and colorscale

    Returns:
        fig (plotly graph object): brain map plot with electrodes added
    """
    print(f"Plotting {len(df)} Electrodes with {args.brain_type} coordinates")
    if args.brain_type == "ave":
        coor_type = "MNI"
    elif args.brain_type == "ind":
        coor_type = "T1"
        assert len(df.subject.unique()) == 1  # only 1 subject
    r = 1.5
    colorbar_show = True
    if args.final or args.final2:
        colorbar_show = False
    if args.final2:
        r = 1.8
    for elecname, center_x, center_y, center_z, effect in zip(
        df.electrode,
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
                name=elecname,
                legendgroup=cbar.title,
                colorscale=cbar.colorscale,
                showscale=colorbar_show,
            )
        )
        colorbar_show = False
    if args.shiny:  # add light to elecs
        fig.update_traces(lighting_specular=0.4)

    return fig


def scale_colorbar(fig, df, cbar, bar_count):
    """Scale colorbar on figure

    Args:
        fig (plotly graph object): brain map plot
        df (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object with title, min and max of bar
        bar_count (int): Colorbar object count

    Returns:
        fig (plotly graph object): brain map plot with electrodes added
    """
    if cbar.bar_min == "min":
        cmin = df["effect"].min()
    else:
        cmin = cbar.bar_min

    if cbar.bar_max == "max":
        cmax = df["effect"].max()
    else:
        cmax = cbar.bar_max
    # if bar_count > 0:
    # fig.update_traces(colorbar_x=1 + 0.2 * bar_count)
    fig.update_traces(
        cmin=cmin,
        cmax=cmax,
        colorbar_x=1 + 0.2 * bar_count,
        colorbar_title=cbar.title,
        colorbar_title_font_size=40,
        colorbar_title_side="right",
    )

    return fig


def filter_df(df_plot, cbar, color_split):
    """Select electrodes in the Colorbar value range to plot

    Args:
        df_plot (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object
        color_split (list): list of Colorbar and limits

    Returns:
        df_colorscale (DataFrame): dataframe with correct rows
    """
    if color_split.index(cbar) == 0:
        min = df_plot.effect.min()
    else:
        try:
            min = float(color_split[color_split.index(cbar) - 1])
        except:
            min = df_plot.effect.min()
    df_colorscale = df_plot.loc[df_plot.effect >= min, :]
    try:
        max = float(color_split[color_split.index(cbar) + 1])
    except:
        max = df_plot.effect.max()
    df_colorscale = df_colorscale.loc[df_colorscale.effect <= max, :]
    return df_colorscale


def update_properties(fig, args):
    """Adjust camera angle and lighting for fig

    Args:
        fig (plotly graph object): brain map plot

    Returns:
        fig (plotly graph object): brain map plot with correct angle and lighting
    """
    # camera = dict(  # Old view
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=-1.5, y=0, z=0),
    # )
    if args.final:
        camera = dict(  # New2 view
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.5, y=0, z=-0.1),
        )
    elif args.final2:
        camera = dict(  # New view
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-10, y=-0.15, z=-0.1),
        )
    else:
        camera = dict(  # Zaid's view
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.5, y=0.2, z=0),
        )

    if args.final:
        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=0.5, y=0.8, z=0.6),
        )
    else:
        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )

    if args.final2:
        roughness = 0.3
    else:
        roughness = 0.4

    fig.update_traces(
        colorbar_thickness=40,
        colorbar_tickfont_size=30,
        lighting_roughness=roughness,
        lightposition=dict(x=0, y=0, z=100),
    )
    fig.update_layout(
        scene_camera=camera, scene=scene, margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    return fig


def plot_brainmap(args, df, outfile):
    """Plot brainmap plot given a df file with coordinates and effects

    Args:
        df (pandas DataFrame): DataFrame with electrode coordinates and effects
        outfile (str): outfile name

    Returns:
        fig (plotly graph object): brain map plot
    """
    if len(df) == 0:
        print("Empty Dataframe")
        return
    # Plot Brain Surface
    surf1, surf2 = load_surf(args)
    fig = go.Figure()
    if surf1:  # left hemisphere
        fig = plot_surf(fig, surf1, args)
    if surf2:  # right hemisphere
        fig = plot_surf(fig, surf2, args)

    # Plot Electrodes and Colorbars
    bar_count = 0
    for cbar in args.color_split:  # Loop through Colorbars
        if not isinstance(cbar, Colorbar):
            continue
        else:  # filter for the values in range
            df_colorscale = filter_df(df, cbar, args.color_split)

        fignew = go.Figure()
        fignew = plot_electrodes(fignew, df_colorscale, cbar, args)
        fignew = scale_colorbar(fignew, df_colorscale, cbar, bar_count)
        bar_count += 1

        # Add electrode traces to main figure
        for trace in range(0, len(fignew.data)):
            fig.add_trace(fignew.data[trace])

    # Save Plot
    fig = update_properties(fig, args)
    if len(outfile) > 0:
        width = 1200 + 100 * bar_count
        if args.final or args.final2:
            width = 1000
        try:
            print(f"Writing to {outfile}")
            # fig_bytes = fig.to_image(format="png", scale=100, width=width, height=1000)
            # buf = io.BytesIO(fig_bytes)
            # img = Image.open(buf)
            # breakpoint()
            fig.write_image(outfile, scale=10, width=width, height=1000)
        except:
            print("File format not provided/supported")
    else:
        print("Not writing to file")
    return fig


def make_brainmap(args, df, outfile=""):
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
        df.loc[df.subject == "7170", "subject"] = "717"  # fix for 717

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
    fig = plot_brainmap(args, df_plot, outfile=outfile)
    return fig


class Colorbar(object):
    """Colorbar Object

    Args:
        title (str): Colorbar title
        colorscale (colorscale): plotly colorscale (https://plotly.com/python/builtin-colorscales/)
        bar_min (str/int): minimum of the bar
        bar_max (str/int): maximum of the bar

    Returns:
        colorbar (Colorbar)
    """

    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "correlation")
        self.colorscale = kwargs.get(
            "colorscale", [[0, "rgb(255,0,0)"], [1, "rgb(255,255,0)"]]
        )
        self.bar_min = kwargs.get("bar_min", "min")
        self.bar_max = kwargs.get("bar_max", "max")

    def __repr__(self):
        return f"title: {self.title}, colorscale: {self.colorscale}, bar_min: {self.bar_min}, bar_max: {self.bar_max}"


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Aggregate data
    df = aggregate_data(args)
    df = organize_data(args, df)
    df = add_effect(args, df)

    # Plot brainmap for comp/prod
    for key in args.keys:
        make_brainmap(
            args,
            df.loc[df.key == key, ("electrode", "effect")],
            args.outfile % key,
        )
    return


if __name__ == "__main__":
    main()
