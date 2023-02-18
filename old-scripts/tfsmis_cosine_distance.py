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
from tfsmis_word_gap import add_word_info


def run_pca(df):
    pca = PCA(n_components=50, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


def get_dist(arr1, arr2):
    return distance.cosine(arr1, arr2)


def get_freq(word):
    return word_frequency(word, "en")


def plot_dist(df, sid, col1, col2, suffix=""):
    fig, ax = plt.subplots(figsize=(15, 6))
    # ax.set(yscale="log")
    ax = sns.scatterplot(data=df, x=col1, y=col2)
    linefit = stats.linregress(df[col1], df[col2])
    # ax.plot(
    #     df[col1],
    #     df[col1] * linefit.slope + linefit.intercept,
    #     c="r",
    # )
    # corr = stats.pearsonr(df[col1], df[col2])
    # ax.text(
    #     0.9,
    #     ax.get_ylim()[1],
    #     f"r={corr[0]:3f} p={corr[1]:.3f}",
    #     color="r",
    # )
    plot_name = col1 + "_vs_" + col2
    plt.savefig("results/figures/" + sid + "_" + plot_name + suffix)
    plt.close()


def plot_hist(df, sid, col1, suffix="_hist"):
    fig, ax = plt.subplots(figsize=(15, 6))
    # ax.set(yscale="log")
    ax = sns.histplot(data=df, x=col1)
    plt.savefig("results/figures/" + sid + "_" + col1 + suffix)
    plt.close()


def add_cosine_info(datum):

    datum["embeddings_n"] = datum.embeddings.shift(-1)
    datum["embeddings_n-2"] = datum.embeddings.shift(1)
    datum = datum[
        datum.conversation_id.shift(-1) == datum.conversation_id.shift(1)
    ]

    datum["cos-dist_nn-1"] = datum.apply(
        lambda x: get_dist(x["embeddings_n"], x["embeddings"]), axis=1
    )
    datum["cos-dist_nn-2"] = datum.apply(
        lambda x: get_dist(x["embeddings_n-2"], x["embeddings_n"]), axis=1
    )

    datum["word_freq_en"] = datum.apply(lambda x: get_freq(x["word"]), axis=1)

    return datum


def plot_cosine_info(df, sid):
    plot_hist(df, sid, "cos-dist_nn-1")
    plot_hist(df, sid, "cos-dist_nn-2")
    plot_dist(df, sid, "cos-dist_nn-1", "true_pred_prob")
    plot_dist(df, sid, "cos-dist_nn-2", "true_pred_prob")
    plot_dist(df, sid, "cos-dist_nn-1", "top1_pred_prob")
    plot_dist(df, sid, "cos-dist_nn-2", "top1_pred_prob")
    plot_dist(df, sid, "cos-dist_nn-1", "word_freq_en")
    plot_dist(df, sid, "cos-dist_nn-2", "word_freq_en")
    plot_dist(df, sid, "cos-dist_nn-1", "word_freq_phase")
    plot_dist(df, sid, "cos-dist_nn-2", "word_freq_phase")


def plot_word_info(df, sid):
    df_comp = df.loc[df.production == 0]
    df_prod = df.loc[df.production == 1]

    # plot_hist(df_comp, sid, "word_len", "_comp")
    # plot_hist(df_prod, sid, "word_len", "_prod")
    plot_hist(df_comp, sid, "word_lengap", "_comp")
    plot_hist(df_prod, sid, "word_lengap", "_prod")
    # plot_dist(df_comp, sid, "true_pred_prob", "word_gap", "_comp")
    # plot_dist(df_prod, sid, "true_pred_prob", "word_gap", "_prod")
    # plot_dist(df_comp, sid, "top1_pred_prob", "word_gap", "_comp")
    # plot_dist(df_prod, sid, "top1_pred_prob", "word_gap", "_prod")


@main_timer
def main():

    sid = "625"
    file_name = (
        "data/tfs/"
        + sid
        + "/pickles/"
        + sid
        + "_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl"
    )
    df = load_datum(file_name)
    print(f"After loading: Datum loads with {len(df)} words")
    df = clean_datum("gpt2-xl", df, False)
    df = run_pca(df)

    # df = add_cosine_info(df)
    # plot_cosine_info(df, sid)
    df = add_word_info(df, "word_len", "utt")

    breakpoint()
    plot_word_info(df, sid)

    return


if __name__ == "__main__":
    main()
