import os
import string

import numpy as np
import pandas as pd
import string
from scipy.spatial import distance
from scipy import stats
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordfreq import word_frequency

from utils import main_timer
from tfsenc_read_datum import load_datum, clean_datum
from tfsmis_cosine_distance import get_dist, plot_hist, plot_dist

import gensim.downloader as api
import re


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def plot_hist(df, mode):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax = sns.histplot(data=df, x="utt_count")

    ax.set_yscale("log")
    plot_name = "utt_len_" + mode
    plt.savefig("results/figures/717_" + plot_name)
    plt.close()


def plot_hist_all(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax = sns.histplot(data=df, x="utt_count", hue="utt")

    ax.set_yscale("log")
    plot_name = "utt_len"
    plt.savefig("results/figures/717_" + plot_name)
    plt.close()


@main_timer
def main():
    sid = "798"
    file_name = (
        "/home/kw1166/scratch/247-encoding/data/tfs/"
        + sid
        + "/pickles/"
        + sid
        + "_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl"
    )

    df = load_datum(file_name)
    print(f"After loading: Datum loads with {len(df)} words")
    df = clean_datum("gpt2", df, False)
    print(f"After cleaning: Datum now has {len(df)} words")
    df = df[~df.duplicated(subset=["word", "adjusted_onset"])]
    print(f"Removing duplicated words. Datum now has {len(df)} words")
    breakpoint()

    # df.loc[
    #     df.duplicated(subset=["word", "adjusted_onset"], keep=False),
    #     (
    #         "word",
    #         "onset",
    #         "adjusted_onset",
    #         "sentence",
    #         "token",
    #         "conversation_name",
    #     ),
    # ]
    # breakpoint()

    df2 = df.loc[
        :,
        (
            "index",
            "word",
            "production",
            "conversation_id",
            "conversation_name",
            "onset",
            "offset",
            "adjusted_onset",
            "adjusted_offset",
            "top1_pred",
            "top1_pred_prob",
            "true_pred_prob",
            "surprise",
            "entropy",
            "embeddings",
        ),
    ]
    df2.to_csv(sid + "_preds.csv")
    breakpoint()

    # content words
    glove = api.load("glove-wiki-gigaword-50")
    df["emb_actual"] = df.word.str.strip().apply(
        lambda x: get_vector(x.lower(), glove)
    )
    df = df[df.emb_actual.notna()]

    df["emb_counterfactual"] = df.top1_pred.str.strip().apply(
        lambda x: get_vector(x.lower(), glove)
    )
    df = df[df.emb_counterfactual.notna()]

    df.loc[:, "emb_dist"] = df.apply(
        lambda x: get_dist(x["emb_actual"], x["emb_counterfactual"]), axis=1
    )

    df_output = df.loc[:, ["word", "top1_pred", "emb_dist"]]
    df_output["word"] = df_output.word.str.strip()
    df_output["top1_pred"] = df_output.top1_pred.str.strip()
    df_output.to_csv("content_words_glove.csv")

    # utterance length
    df["utt"] = 0
    df.loc[df["speaker"] == "Speaker1", "utt"] = 1

    df["utt_count"] = (
        df.groupby(df["utt"].ne(df["utt"].shift()).cumsum()).cumcount().add(1)
    )

    df_shift = df[df["utt"].ne(df["utt"].shift(-1))]

    df_comp = df_shift[df_shift["utt"] == 0]
    df_prod = df_shift[df_shift["utt"] == 1]

    plot_hist(df_comp, "comp")
    plot_hist(df_prod, "prod")
    plot_hist_all(df_shift)

    return


if __name__ == "__main__":
    main()
