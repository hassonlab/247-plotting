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

    sig_elecs = {}
    areas = {
        # "allnodepth": "",
        "IFG": "-ifg",
        "aSTG": "-astg",
        "mSTG": "-mstg",
        "SM": "-sm",
        "STG": "-stg",
    }
    embs = ["whisper-en-max", "whisper-de-max", "gpt2-medium"]

    for _, val in areas.items():
        for emb in embs:
            sig_key = f"{emb}{val}"
            sig_path = f"data/plotting/ccn-sig-file-{sig_key}.csv"
            sig_file = pd.read_csv(sig_path)
            sig_file["elec"] = (
                sig_file.subject.astype(str) + "_" + sig_file.electrode
            )
            sig_elecs[sig_key] = sig_file.elec.tolist()

    # pdf_name = "max-cor-layers-max-118-aligned.pdf"
    # pdf = PdfPages(pdf_name)
    # SMALL_SIZE = 20
    # plt.rc("font", size=SMALL_SIZE)
    # plt.rc("axes", titlesize=SMALL_SIZE)
    save_df = pd.DataFrame({'Layer':layers})
    plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")
    for area, area_tag in areas.items():
        print(f"{area}")
        cors = []
        for key2, _ in format.items():
            if "whisper-en" in key2:
                key = f"{embs[0]}{area_tag}"
                en_elecs = len(sig_elecs[key])
            elif "whisper-de" in key2:
                key = f"{embs[1]}{area_tag}"
                de_elecs = len(sig_elecs[key])
            elif "gpt2-medium" in key2:
                key = f"{embs[2]}{area_tag}"
                gpt2_elecs = len(sig_elecs[key])
            layer = int(key2[-2:])
            cors = read_folder(
                cors, format[key2], sig_elecs, key, label=layer, type=key
            )
        cors = pd.concat(cors)
        cors.rename(columns={"label": "Layer"}, inplace=True)
        layer_max = cors.groupby(["type", "Layer"]).mean().max(axis=1)

        cmap = plt.cm.get_cmap("winter")
        colors = [cmap(i / len(layers)) for i in range(0, len(layers))]
        # colors = [colorFader('#031c47','#b3cefc',i/len(layers)) for i in range(0,len(layers))]

        assert len(layer_max == 75)
        medium_layers = layer_max[:25].tolist()
        de_layers = layer_max[25:50].tolist()
        en_layers = layer_max[50:].tolist()
        save_df[f"{area}-medium"] = medium_layers
        save_df[f"{area}-de"] = de_layers
        save_df[f"{area}-en"] = en_layers
        fig, ax = plt.subplots()
        ax.scatter(
            layers,
            de_layers,
            s=20,
            facecolors="none",
            edgecolors="grey",
            marker="o",
        )
        ax.scatter(
            layers,
            medium_layers,
            s=20,
            marker="o",
            color='grey',
            # facecolors="none",
            # edgecolors="grey",
        )
        ax.scatter(layers, en_layers, s=20, color=colors, marker="o")
        model = LinearRegression().fit(layers[1:].reshape((-1, 1)),en_layers[1:])
        slope = model.coef_[0]
        print("Slope", slope * 100)
        ax.set(
            xlabel="Layers",
            ylabel="Max Correlation (r)",
            # title=f"{area}(en:{en_elecs},de:{de_elecs},gpt2:{gpt2_elecs})",
            title=f"{area}(encoder:{en_elecs},gpt2:{gpt2_elecs},slope:{slope})",
        )
        # if "-aligned" in pdf_name:
            # ax.set_ylim(0,0.34)
        ax.set_ylim(0,0.45)
        # pdf.savefig(fig)
        plt.savefig(f"winter_{area}.svg")
        plt.close()
    breakpoint()
    # pdf.close()
    return


if __name__ == "__main__":
    main()
