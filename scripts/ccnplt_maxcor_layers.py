import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from tfsplt_utils import read_folder


def main():

    ##### Encoding Results Folder #####
    layers = np.arange(0, 25)

    format = {}
    for layer in layers:
        format[
            f"whisper-en-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-encoder-new-lag5k-25-all-{layer}/*/*_comp.csv"
        format[
            f"whisper-de-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-decoder-lag5k-25-all-{layer}/*/*_comp.csv"
        format[
            f"gpt2-medium-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-gpt2-medium-lag5k-25-all-shift-emb-118-{layer}/*/*_comp.csv"

    df = pd.read_csv("data/plotting/elec_masterlist.csv")
    df.dropna(subset="subject", inplace=True)
    df["subject"] = df.subject.astype(int).astype(str)
    df["elec"] = df.subject + "_" + df.name

    df_160 = df[df.part160 == 1]  # sig(160)

    # Choose which elec list to use
    final_list = df_160

    sig_elecs = {}
    areas = {
        "all": ["whole"],
        "allnodepth": ["whole-nodepth"],
        "IFG": ["IFG"],
        "STG": ["STG"],
        "SM": ["precentral", "postcentral", "premotor", "postcg"],
        "TP": ["TP"],
        "aMTG": ["aMTG"],
        "pmtg": ["pmtg"],
        "AG": ["AG"],
        "SMG": ["parietal"],
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

    pdf = PdfPages("results/max-cor-layers-gpt-118.pdf")
    SMALL_SIZE = 20
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    for key, val in sig_elecs.items():
        cors = []
        for key2, _ in format.items():
            layer = int(key2[-2:])
            type = key2[8:10]
            cors = read_folder(
                cors, format[key2], sig_elecs, key, label=layer, type=type
            )
        cors = pd.concat(cors)
        cors.rename(columns={"label": "Layer"}, inplace=True)
        layer_max = cors.groupby(["type", "Layer"]).mean().max(axis=1)

        assert len(layer_max == 75)
        de_layers = layer_max[:25].tolist()
        en_layers = layer_max[25:50].tolist()
        medium_layers = layer_max[50:].tolist()

        fig, ax = plt.subplots(figsize=(18, 10))
        ax.scatter(layers, en_layers, s=50, color="red")
        ax.scatter(layers, de_layers, s=50, color="blue")
        ax.scatter(layers, medium_layers, s=50, color="green")
        ax.set(
            xlabel="Layers",
            ylabel="Max Correlation (r)",
            title=f"{key}({len(val)})",
        )
        pdf.savefig(fig)
        plt.close()

    pdf.close()
    return


if __name__ == "__main__":
    main()
