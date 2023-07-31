import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from tfsplt_utils import read_folder, colorFader


def main():
    ##### Encoding Results Folder #####
    layers = np.arange(0, 25)

    lags = np.arange(-5000, 5025, 25)
    select_lags = np.arange(-2000, 2025, 25)
    select_idx = [idx for idx, val in enumerate(lags) if val in select_lags]

    format = {}
    for layer in layers:
        format[
            f"whisper-en-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-encoder-new-lag5k-25-all-{layer}/*/*_comp.csv"

    sig_elecs = {}
    areas = {
        # "allnodepth": "",
        "IFG": "-ifg",
        "aSTG": "-astg",
        "mSTG": "-mstg",
        "SM": "-sm",
        "STG": "-stg",
    }
    embs = ["whisper-en-max"]

    for _, val in areas.items():
        for emb in embs:
            sig_key = f"{emb}{val}"
            sig_path = f"data/plotting/ccn-sig-file-{sig_key}.csv"
            sig_file = pd.read_csv(sig_path)
            sig_file["elec"] = sig_file.subject.astype(str) + "_" + sig_file.electrode
            sig_elecs[sig_key] = sig_file.elec.tolist()

    plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")
    for area, area_tag in areas.items():
        print(f"{area}")
        cors = []
        for key2, _ in format.items():
            key = f"{embs[0]}{area_tag}"
            en_elecs = len(sig_elecs[key])

            layer = int(key2[-2:])
            cors = read_folder(
                cors, format[key2], sig_elecs, key, label=layer, type=key
            )
        cors = pd.concat(cors)
        cors.rename(columns={"label": "Layer"}, inplace=True)
        layer_max = cors.groupby(["type", "Layer"]).mean()
        layer_max = layer_max.loc[:, select_idx]

        cmap = plt.cm.get_cmap("winter")
        colors = [cmap(i / len(layers)) for i in range(0, len(layers))]
        # colors = [
        #     colorFader("#031c47", "#b3cefc", i / len(layers))
        #     for i in range(0, len(layers))
        # ]

        fig, ax = plt.subplots()
        plot_lags = select_lags / 1000
        extratick = plot_lags[layer_max.max().argmax()]

        for layer in layers:
            ax.plot(plot_lags, layer_max.iloc[layer, :], color=colors[layer])

        ax.set(
            xlabel="Lags(s)",
            ylabel="Correlation (r)",
            title=f"{area} ({en_elecs})",
        )

        ax.set_ylim(0, 0.45)
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_yticklabels([0, 0.2, 0.4])
        ax.set_xticks([-2, -1, 0, extratick, 1, 2])
        ax.set_xticklabels([-2, -1, 0, extratick, 1, 2])
        ax.xaxis.get_ticklines()[6].set_markeredgecolor("lime")
        plt.setp(ax.get_xticklabels()[3], color="lime")

        plt.savefig(f"enc_{area}.svg")
        plt.close()

    return


if __name__ == "__main__":
    main()
