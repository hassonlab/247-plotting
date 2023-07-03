import glob
import os
import socket

import numpy as np
import pandas as pd
from scipy.io import loadmat

import plotly.io as pio
import plotly.graph_objects as go


# loading brain surface plot
def load_surf(path, id, hemisphere="left"):
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


# plot 3D brain
def plot_brain(fig, surf):
    # surf["faces"] is an n x 3 matrix of indices into surf["coords"]; connectivity matrix
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


# plot 3D electrodes
def plot_electrodes(fig, df, colorscale, cbar_title="correlations"):
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
                legendgroup=cbar_title,
                colorscale=colorscale,
            )
        )

    return fig


# reading electrode coordinates
def read_coor(path, id):
    df_coor = pd.DataFrame()
    for sid in id:
        file = os.path.join(path, f"{sid}", f"{sid}-electrode-coordinates.csv")
        df = pd.read_csv(file)
        df.dropna(subset=["name"], inplace=True)
        df["subject"] = sid
        df_coor = pd.concat([df_coor, df])

    return df_coor


# Set min/max of colorbar
def scale_colorbar(
    fig, df, bar_count, cbar_title="correlation", cbar_min="min", cbar_max="max"
):
    if cbar_min == "min":
        cmin = df["effect"].min()
    else:
        cmin = cbar_min

    if cbar_max == "max":
        cmax = df["effect"].max()
    else:
        cmax = cbar_max
    if bar_count > 0:
        fig.update_traces(colorbar_x=1 + 0.2 * bar_count)
    fig.update_traces(
        cmin=cmin,
        cmax=cmax,
        colorbar_title=cbar_title,
        colorbar_title_font_size=40,
        colorbar_title_side="right",
    )

    return fig


def update_properties(fig):
    # Left hemisphere
    # TODO: add camera for other views
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


def plot_brainmap(df, outfile, color_split=[]):
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
        fig = plot_brain(fig, surf1)
    if surf2:  # right hemisphere
        fig = plot_brain(fig, surf2)

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
        fignew = plot_electrodes(
            fignew,
            df_colorscale,
            cbar["colorscale"],
            cbar["title"],
        )
        fignew = scale_colorbar(
            fignew,
            df_colorscale,
            bar_count,
            cbar_title=cbar["title"],
            cbar_min=cbar["min"],
            cbar_max=cbar["max"],
        )
        bar_count += 1

        # Add electrode traces to main figure
        for trace in range(0, len(fignew.data)):
            fig.add_trace(fignew.data[trace])

    # Save Plot
    fig = update_properties(fig)
    fig.write_image(outfile, scale=6, width=1200 + 100 * bar_count, height=1000)

    return


def main():
    df = pd.read_csv("eff_file.csv")
    df.drop(columns={"label", "sid"}, inplace=True)
    df.set_index(["electrode"], inplace=True)

    df_comp = df.loc[df.key == "prod", :].max(axis=1, numeric_only=True)

    col1 = {
        "title": "correlation1",
        "colorscale": [[0, "rgb(255,248,240)"], [1, "rgb(255,0,0)"]],
        "min": "min",
        "max": "max",
    }
    col2 = {
        "title": "correlation2",
        "colorscale": "viridis_r",
        "min": "min",
        "max": "max",
    }
    color_split = [col1, 0.1, col2]
    color_split = [col2]
    outfile = "fig.png"

    plot_brainmap(df_comp, outfile, color_split)

    return


if __name__ == "__main__":
    main()
