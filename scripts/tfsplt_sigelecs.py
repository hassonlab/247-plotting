import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import plot_markers
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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


def organize_data_elec_sig(args, df, variable):
    df.columns = [
        "label",
        "electrode",
        "key",
        "sid",
        "pr",
        "pca95",
        "pca99",
        "lags",
        "samples",
    ]
    # df["pr_ratio"] = df.pr / df.samples * 1000
    # df["pca99_ratio"] = df.pca99 / df.samples
    melted_df = pd.melt(df, id_vars=["electrode", "key", "label", "sid"])
    new_df = melted_df[0::2]
    new_df["value2"] = melted_df[1::2].value.tolist()
    df_plot = new_df[new_df["variable"] == variable]

    # def rgb_to_hex(color):
    #     return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

    # cc = Colormap2D(
    #     args.cmap,
    #     vmin=df_plot["value"].min(),
    #     vmax=df_plot["value"].max(),
    #     vmin2=df_plot["value2"].min(),
    #     vmax2=df_plot["value2"].max(),
    #     vflip=False,
    #     hflip=False,
    # )
    # red, green, blue, alpha = cc(df_plot.loc[:, ("value", "value2")].to_numpy())
    # colors = np.vstack((red, green, blue, alpha)).T
    # colors_hex = [rgb_to_hex(color) for color in colors]
    # df_plot["effect"] = colors_hex
    # print(df_plot.value.describe())
    # print(df_plot.value2.describe())

    # df_plot.reset_index(inplace=True)

    return df_plot


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Get effect
    sigelec = [args.formats[0]]
    args.formats = [args.formats[1]]
    df = aggregate_data(args)
    df = organize_data(args, df)
    df["effect"] = df.max(axis=1)
    df.reset_index(inplace=True)

    args.formats = sigelec
    df2 = aggregate_data(args, take_last=False)
    df2 = organize_data_elec_sig(args, df2, "pr")
    df2["effect"] = df.effect.tolist()

    # df3["effect"] = df.effect.tolist()
    # df3 = organize_data_elec_sig(args, df2, "pr")
    # df4 = organize_data_elec_sig(args, df2, "pca95")
    # df5 = organize_data_elec_sig(args, df2, "pca99")
    # df3["pca95_1"] = df4.value.tolist()
    # df3["pca95_2"] = df4.value2.tolist()
    # df3["pca99_1"] = df5.value.tolist()
    # df3["pca99_2"] = df5.value2.tolist()
    # X = df3.iloc[:, 5:7].to_numpy()
    # Y = df3.iloc[:, -1].to_numpy()
    # model = make_pipeline(StandardScaler(), LinearRegression())
    # model.fit(X, Y)
    # model.score(X, Y)

    sns.set_style("whitegrid")

    for key in args.keys:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.scatterplot(
            data=df2.loc[df2.key == key, :],
            x="value",
            y="value2",
            hue="effect",
            # size="effect",
            palette=args.cmap,
            # edgecolor=None,
        )
        plt.savefig(args.outfile % key)

    return


if __name__ == "__main__":
    main()
