import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from tfsplt_utils import read_folder


def main():

    ##### Encoding Results Folder #####
    layers = np.arange(0, 25)
    layers = [1, 24]
    lags = np.arange(-5000, 5025, 25)
    select_lags = np.arange(-2000, 2025, 25)
    select_idx = [idx for idx, val in enumerate(lags) if val in select_lags]

    format = {}
    for layer in layers:
        format[
            f"whisper-en-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-encoder-new-lag5k-25-all-{layer}/*/*_comp.csv"
        format[
            "whisper-en-00"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-encoder-concat-lag5k-25-all/*/*_comp.csv"

    sig_elecs = {}
    areas = {
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
            sig_file["elec"] = (
                sig_file.subject.astype(str) + "_" + sig_file.electrode
            )
            sig_elecs[sig_key] = sig_file.elec.tolist()

    plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")
    for area, area_tag in areas.items():
        fig, ax = plt.subplots()
        print(f"{area}")
        cors = []
        for key2, _ in format.items():
            key = f"{embs[0]}{area_tag}"
            layer = int(key2[-2:])
            cors = read_folder(
                cors, format[key2], sig_elecs, key, label=layer, type=key
            )
        cors = pd.concat(cors)
        cors.rename(columns={"label": "Layer"}, inplace=True)

        layer_max = cors.groupby(["type", "Layer"]).mean()
        layer_max = layer_max.loc[:, select_idx]

        # variance partitioning
        sgn_variance_01 = layer_max.iloc[1, :] ** 2 * np.sign(
            layer_max.iloc[1, :]
        )
        sgn_variance_24 = layer_max.iloc[-1, :] ** 2 * np.sign(
            layer_max.iloc[-1, :]
        )
        sgn_variance_concat = layer_max.iloc[0, :] ** 2 * np.sign(
            layer_max.iloc[0, :]
        )
        sgn_variance_common = (
            sgn_variance_01 + sgn_variance_24 - sgn_variance_concat
        )
        sgn_variance_01_unique = sgn_variance_01 - sgn_variance_common
        sgn_variance_24_unique = sgn_variance_24 - sgn_variance_common
        sgn_variance_common[sgn_variance_common < 0] = 0
        sgn_variance_01_unique[sgn_variance_01_unique < 0] = 0
        sgn_variance_24_unique[sgn_variance_24_unique < 0] = 0

        cor_common = np.sqrt(sgn_variance_common)
        cor_01 = np.sqrt(sgn_variance_01_unique)
        cor_24 = np.sqrt(sgn_variance_24_unique)

        vals = [max(cor_01), max(cor_common), max(cor_24)]
        cat = ["Layer 1\nUnique", "Shared", "Layer 24\nUnique"]
        ax.bar(cat, vals, color=["red", "#636cad", "#48c279"])
        print(vals)

        # ax.plot(
        #     select_lags,
        #     cor_common,
        #     color="green",
        # )
        # ax.plot(
        #     select_lags,
        #     cor_01,
        #     color="red",
        # )
        # ax.plot(
        #     select_lags,
        #     cor_24,
        #     color="blue",
        # )

        ax.set(
            # xlabel="Lags(ms)",
            ylabel="Avg. Partial Correlation",
            # title=f"mstg:orange,astg:blue,ifg:red,sm:green",
        )
        ax.set_ylim(0, 0.38)
        plt.savefig(f"var_par_{area}.svg")
        plt.close()

    return


if __name__ == "__main__":
    main()
