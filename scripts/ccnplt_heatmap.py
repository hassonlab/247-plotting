import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tfsplt_utils import read_folder


def plot_heatmap(data, key, val, vmin=-100, vmax=-100):
    xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    xticklabels = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    plt.figure(figsize=(18, 10))

    if vmax == -100:
        ax = sns.heatmap(data, cmap="crest")
    else:
        ax = sns.heatmap(data, cmap="crest", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Lag (s)")
    ax.set_title(f"{key} ({len(val)})")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    if vmax == -100:
        plt.savefig(f"{key}.png")
    else:
        plt.savefig(f"{key}_aligned.png")

    return


def main():
    lags = np.arange(-5000, 5025, 25)
    select_lags = np.arange(-2000, 2025, 25)

    ##### Encoding Results Folder #####
    layers = np.arange(0, 25)

    # emb_name = "whisper-decoder"
    # emb_folder = "whisper-medium.en-decoder"

    # emb_name = "whisper-encoder"
    # emb_folder = "whisper-medium.en-encoder-new"

    emb_name = "gpt2-medium"
    emb_folder = "gpt2-medium"

    format = {}
    for layer in layers:
        format[
            f"{emb_name}-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-{emb_folder}-lag5k-25-all-shift-emb-118-{layer}/*/*_comp.csv"

    # Output directory name
    # OUTPUT_DIR = "results/podcast-heatmaps-cnn"

    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)

    df = pd.read_csv("data/plotting/elec_masterlist.csv")
    df.dropna(subset="subject", inplace=True)
    df["subject"] = df.subject.astype(int).astype(str)
    df["elec"] = df.subject + "_" + df.name

    df_160 = df[df.part160 == 1]  # sig(160)

    # Choose which elec list to use
    final_list = df_160
    # final_list = df

    sig_elecs = {}
    areas = {
        "IFG": ["IFG"],
        "STG": ["STG"],
        "SM": ["precentral", "postcentral", "premotor", "postcg"],
        "TP": ["TP"],
        "aMTG": ["aMTG"],
        "pmtg": ["pmtg"],
        "AG": ["AG"],
        "SMG": ["parietal"],
        "all": ["whole"],
        "allnodepth": ["whole-nodepth"],
    }

    for key, val in areas.items():
        area_elecs = []
        for sub_area in val:
            if sub_area == "whole":  # whole brain
                area_elecs.extend(final_list.loc[:, "elec"].tolist())
            elif sub_area == "whole-nodepth":  # whole brain no depth
                area_elecs.extend(
                    final_list.loc[final_list.type != "D", "elec"].tolist()
                )
            else:
                area_elecs.extend(
                    final_list.loc[
                        final_list.princeton_class == sub_area, "elec"
                    ].tolist()
                )
        sig_elecs[key] = area_elecs

    for key, val in sig_elecs.items():
        print(f"{key}({len(val)})")
        cors = []
        for key2, _ in format.items():
            layer = int(key2[-2:])
            cors = read_folder(
                cors, format[key2], sig_elecs, key, label=layer, type=key
            )
        cors = pd.concat(cors)
        cors.rename(columns={"label": "Layer"}, inplace=True)
        ave_cors = cors.groupby("Layer").mean()

        select_idx = [idx for idx, value in enumerate(lags) if value in select_lags]
        ave_cors = ave_cors.loc[:, select_idx]
        ave_cors.columns = select_lags

        SMALL_SIZE = 20
        plt.rc("font", size=SMALL_SIZE)
        plt.rc("axes", titlesize=SMALL_SIZE)
        plot_heatmap(ave_cors, key, val)
        plot_heatmap(ave_cors, key, val, 0, 0.35)

    return


if __name__ == "__main__":
    main()
