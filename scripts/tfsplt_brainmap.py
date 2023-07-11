import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

import plotly.graph_objects as go


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


def load_surf(path, id, hemisphere="left"):
    """Load brain surface mat files

    Args:
        path (str): brain surface mat file directory
        id (list): list of subject id
        hemisphere (str): hemisphere to load, either left, right, or both

    Returns:
        surf1 (dict): brain surface mat
        surf2 (dict): brain surface mat
    """

    if len(id) > 1:
        file = glob.glob(os.path.join(path, "*.mat"))
    else:
        file = glob.glob(os.path.join(path, id[0], f"NY{id[0]}*.mat"))

    assert len(file) <= 2, "Duplicate surface mat files exists"
    if hemisphere == "left":  # left hemisphere
        surf1 = loadmat(file[0])
        surf2 = None
    elif hemisphere == "right":  # right hemisphere
        surf1 = None
        surf2 = loadmat(file[1])
    else:  # both hemispheres
        surf1 = loadmat(file[0])
        surf2 = loadmat(file[1])

    return surf1, surf2


def plot_surf(fig, surf):
    """Plot brain surface onto figure

    Args:
        fig (plotly graph object): brain map plot
        surf (dict): brain surface file

        Note: surf["faces"] is an n x 3 matrix of indices into surf["coords"]; connectivity matrix

    Returns:
        fig (plotly graph object): brain map plot with brain surface added
    """

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

    fig.update_traces(lighting_ambient=0.3)
    return fig


def plot_electrodes(fig, df, cbar):
    """Plot electrodes onto figure

    Args:
        fig (plotly graph object): brain map plot
        df (DataFrame): dataframe with electrode coordinates and effects
        cbar (Colorbar): Colorbar object with title and colorscale

    Returns:
        fig (plotly graph object): brain map plot with electrodes added
    """
    r = 1.5
    for elecname, center_x, center_y, center_z, effect in zip(
        df.electrode, df.MNI_X, df.MNI_Y, df.MNI_Z, df.effect
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
            )
        )

    return fig


def scale_colorbar(fig, df, cbar, bar_count):
    """Plot and scale colorbar on figure

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


def update_properties(fig):
    """Adjust camera angle and lighting for fig

    Args:
        fig (plotly graph object): brain map plot

    Returns:
        fig (plotly graph object): brain map plot with correct angle and lighting
    """
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=0, z=0),
    )

    scene = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="auto",
    )

    fig.update_layout(scene_camera=camera, scene=scene)
    fig.update_traces(
        lighting_specular=0.4,
        colorbar_thickness=40,
        colorbar_tickfont_size=30,
        lighting_roughness=0.4,
        lightposition=dict(x=0, y=0, z=100),
    )

    return fig


def plot_brainmap(df_plot, main_dir, color_split):
    """Plot brainmap plot given a df file with coordinates and effects

    Args:
        df (pandas DataFrame): DataFrame with electrode coordinates and effects
        main_dir (str): main_dir
        color_split (list): list of Colorbar and limits

    Returns:
        fig (plotly graph object): brain map plot
    """
    # Plot Brain Surface
    subjects = df_plot.subject.unique()
    surf1, surf2 = load_surf(main_dir, subjects)
    fig = go.Figure()
    if surf1:  # left hemisphere
        fig = plot_surf(fig, surf1)
    if surf2:  # right hemisphere
        fig = plot_surf(fig, surf2)

    # Plot Electrodes and Colorbars
    bar_count = 0
    for cbar in color_split:  # Loop through colorbars
        if isinstance(cbar, float):
            continue
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

        fignew = go.Figure()
        fignew = plot_electrodes(fignew, df_colorscale, cbar)
        fignew = scale_colorbar(fignew, df_colorscale, cbar, bar_count)
        bar_count += 1

        # Add electrode traces to main figure
        for trace in range(0, len(fignew.data)):
            fig.add_trace(fignew.data[trace])

    fig = update_properties(fig)
    return fig


def make_brainmap(df, outfile, color_split=[]):
    """Plot and Save brainmap plot given a pandas Series of effects

    Args:
        df (pandas Series): effect indexed by encoding electrode names
        outfile (str): outfile name
        color_split (list): list of Colorbar and limits

    Returns:
        None
    """

    # Reformat Input Series
    print(f"Number of electrodes for encoding: {len(df)}")
    df = df.reset_index()
    df.rename(columns={0: "effect"}, inplace=True)
    df = df.assign(
        subject=df.electrode.str.split("_", n=1, expand=True)[0],
        electrode=df.electrode.str.split("_", n=1, expand=True)[1],
    )
    df.loc[df.subject == "7170", "subject"] = "717"  # fix for 717
    subjects = df.subject.unique()
    # subjects = [sid[0:3] for sid in subjects]  # fix for 717
    main_dir = "data/plotting/brainplot/"

    # Get Electrode Coordinate Files
    df_coor = read_coor(main_dir, subjects)
    df_plot = pd.merge(
        df,
        df_coor,
        how="inner",
        left_on=["subject", "electrode"],
        right_on=["subject", "name"],
    )
    print(f"Number of electrodes for plotting: {len(df_plot)}")

    # Plot Brain Surface
    surf1, surf2 = load_surf(main_dir, subjects)
    fig = go.Figure()
    if surf1:  # left hemisphere
        fig = plot_surf(fig, surf1)
    if surf2:  # right hemisphere
        fig = plot_surf(fig, surf2)

    # Plot Electrodes and Colorbars
    bar_count = 0
    for cbar in color_split:  # Loop through Colorbars
        if isinstance(cbar, float):
            continue
        else:  # filter for the values in range
            df_colorscale = filter_df(df_plot, cbar, color_split)

        fignew = go.Figure()
        fignew = plot_electrodes(fignew, df_colorscale, cbar)
        fignew = scale_colorbar(fignew, df_colorscale, cbar, bar_count)
        bar_count += 1

        # Add electrode traces to main figure
        for trace in range(0, len(fignew.data)):
            fig.add_trace(fignew.data[trace])

    # Save Plot
    fig = update_properties(fig)
    fig.write_image(outfile, scale=6, width=1200 + 100 * bar_count, height=1000)
    # fig.write_html("fig.html") # plotly object

    return


class Colorbar(object):
    """Colorbar Object

    Args:
        title (str): Colorbar title
        colorscale (colorscale): Matplotlib colorscale (can be list of list or str)
        bar_min (str/int): minimum of the bar
        bar_max (str/int): maximum of the bar

    Returns:
        colorbar (Colorbar)
    """

    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "correlation")
        self.colorscale = kwargs.get(
            "colorscale", [[0, "rgb(255,248,240)"], [1, "rgb(255,0,0)"]]
        )
        self.bar_min = kwargs.get("bar_min", "min")
        self.bar_max = kwargs.get("bar_max", "max")

    def __repr__(self):
        return f"Title: {self.title}, Colorscale: {self.colorscale}, Bar Min: {self.bar_min}, Bar Max: {self.bar_max}"


def main():
    df = pd.read_csv("eff_file.csv")
    df.drop(columns={"label", "sid"}, inplace=True)
    df.set_index(["electrode"], inplace=True)

    df_comp = df.loc[df.key == "prod", :].max(axis=1, numeric_only=True)

    col1 = Colorbar()
    col2 = Colorbar(title="correlation2", colorscale="viridis")
    color_split = [col2]
    color_split = [col1, 0.1, col2]
    outfile = "fig.png"

    make_brainmap(df_comp, outfile, color_split)

    return


if __name__ == "__main__":
    main()
