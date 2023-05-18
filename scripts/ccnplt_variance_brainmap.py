import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from tfsplt_utils import read_folder
from ccnplt_brainmap import get_base_df


def main():

    ##### Encoding Results Folder #####
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

    df = get_base_df("777", "ave", [])  # get all electrodes

    # select only sig elecs
    sig_df = pd.read_csv("data/plotting/ccn-sig-file-whisper-en-max.csv")
    sig_df["elec"] = sig_df.subject.astype(str) + "_" + sig_df.electrode
    sig_df.set_index("elec", inplace=True)
    df = df.merge(sig_df, how="inner", left_index=True, right_index=True)
    df.drop(columns=["subject", "electrode"], inplace=True)

    sig_elecs = {}
    sig_elecs["sig"] = df.index.tolist()

    cors = []
    for key2, _ in format.items():
        layer = int(key2[-2:])
        cors = read_folder(
            cors, format[key2], sig_elecs, "sig", label=layer, type="sig"
        )
    cors = pd.concat(cors)
    cors.rename(columns={"label": "Layer"}, inplace=True)

    def get_variance_ratio(layer_max):
        layer_max = layer_max.loc[:, select_idx]
        sgn_variance_concat = layer_max.iloc[0, :] ** 2 * np.sign(
            layer_max.iloc[0, :]
        )
        sgn_variance_01 = layer_max.iloc[1, :] ** 2 * np.sign(
            layer_max.iloc[1, :]
        )
        sgn_variance_24 = layer_max.iloc[2, :] ** 2 * np.sign(
            layer_max.iloc[2, :]
        )
        sgn_variance_common = (
            sgn_variance_01 + sgn_variance_24 - sgn_variance_concat
        )
        # sgn_variance_01_unique = sgn_variance_01 - sgn_variance_common
        sgn_variance_24_unique = sgn_variance_24 - sgn_variance_common
        sgn_variance_common[sgn_variance_common < 0] = 0
        # sgn_variance_01_unique[sgn_variance_01_unique < 0] = 0
        sgn_variance_24_unique[sgn_variance_24_unique < 0] = 0
        cor_common = np.sqrt(sgn_variance_common)
        # cor_01 = np.sqrt(sgn_variance_01_unique)
        cor_24 = np.sqrt(sgn_variance_24_unique)
        return max(cor_24) / max(cor_common)

    cors.sort_values(by=["electrode", "Layer"], inplace=True)
    var_ratio = cors.groupby(["electrode"]).apply(get_variance_ratio)
    df = df.merge(
        var_ratio.rename("var_ratio"), left_index=True, right_index=True
    )
    breakpoint()
    df.loc[:, 0] = df.index
    df = df.loc[:, [0, 1, 2, 3, 4, "var_ratio"]]
    print(df.var_ratio.describe())

    OUTPUT_DIR = "results/podcast-encoder-max"
    sid_file = os.path.join(OUTPUT_DIR, f"777_ave_var_ratio.txt")
    with open(sid_file, "w") as outfile:
        df.to_string(outfile, index=False, header=False)

    df.var_ratio = df.var_ratio / (1 + df.var_ratio)
    print(df.var_ratio.describe())
    sid_file = os.path.join(OUTPUT_DIR, f"777_ave_var_ratio_norm.txt")
    with open(sid_file, "w") as outfile:
        df.to_string(outfile, index=False, header=False)

    return


if __name__ == "__main__":
    main()
