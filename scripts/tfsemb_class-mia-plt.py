import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from tfsplt_utils import Colormap2D
import plotly.express as px


def main():

    # filepath = "results/20240626-mia-classify/classifier_pca50_%s_C%s_L%s.csv"

    # embs = {
    #     "distilgpt2": [1024, 6, "gpt2"],
    #     "gpt2": [1024, 12, "gpt2"],
    #     "gpt2-medium": [1024, 24, "gpt2"],
    #     "gpt2-large": [1024, 32, "gpt2"],
    #     "gpt2-xl": [1024, 48, "gpt2"],
    #     "gpt-neo-125M": [2048, 12, "gpt-neo"],
    #     "gpt-neo-1.3B": [2048, 24, "gpt-neo"],
    #     "gpt-neo-2.7B": [2048, 32, "gpt-neo"],
    #     "gpt-neox-20b": [2048, 44, "gpt-neo"],
    #     "opt-125m": [2048, 12, "opt"],
    #     "opt-350m": [2048, 24, "opt"],
    #     "opt-1.3b": [2048, 24, "opt"],
    #     "opt-2.7b": [2048, 32, "opt"],
    #     "opt-6.7b": [2048, 32, "opt"],
    #     "opt-13b": [2048, 40, "opt"],
    #     "opt-30b-q": [2048, 48, "opt"],
    #     "opt-66b-q": [2048, 64, "opt"],
    #     "Llama-2-7b-hf": [4096, 32, "llama2"],
    #     "Llama-2-13b-hf": [4096, 40, "llama2"],
    #     "Llama-2-70b-hf-q": [4096, 80, "llama2"],
    # }

    # df = pd.DataFrame()
    # for emb in embs:
    #     layers = np.arange(0, embs[emb][1] + 1, 1)
    #     for layer in layers:
    #         csv = filepath % (emb, f"{embs[emb][0]:04d}", f"{layer:02d}")
    #         df_layer = pd.read_csv(csv)
    #         df_layer["emb"] = emb
    #         df_layer["emb_family"] = embs[emb][2]
    #         df_layer["layer"] = layer
    #         df_layer["layer_perc"] = layer / embs[emb][1]
    #         df = pd.concat((df, df_layer))
    # df.columns = [
    #     "class_cat",
    #     "class_num",
    #     "fold_1",
    #     "fold_2",
    #     "fold_3",
    #     "fold_4",
    #     "fold_5",
    #     "fold_6",
    #     "fold_7",
    #     "fold_8",
    #     "fold_9",
    #     "fold_10",
    #     "ave",
    #     "emb",
    #     "emb_family",
    #     "layer",
    #     "layer_perc",
    # ]
    breakpoint()
    filepath = "results/20240626-mia-classify/summary.csv"
    df = pd.read_csv(filepath)

    df_plot = df[df.class_cat == "embeddings-part_of_speech"]
    fig, ax = plt.subplots()
    family_order = ["gpt2", "gpt-neo", "opt", "llama2"]

    # colors = [
    #     sns.color_palette("hls", 8)[3],
    #     sns.color_palette("hls", 8)[6],
    #     sns.color_palette("hls", 8)[4],
    #     sns.color_palette("hls", 8)[7],
    # ]
    # colors = sns.color_palette("Paired", 10)
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df_plot,
        x="layer_perc",
        y="ave",
        units="emb",
        estimator=None,
        hue="emb_family",
        # palette=colors,
        hue_order=["gpt2", "llama2", "opt", "gpt-neo"],
    )
    plt.savefig("class3.jpg")

    return


if __name__ == "__main__":
    main()
