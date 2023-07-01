import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from matplotlib.backends.backend_pdf import PdfPages
from utils import load_pickle


def get_results(df, mode):
    if mode == "word_len":
        word_len = df.adjusted_offset - df.adjusted_onset
    elif mode == "word_gap":
        if df.conversation_id.nunique() > 1:
            word_len = []
            for convo_id in df.conversation_id.unique():
                df_part = df.loc[df["conversation_id"] == convo_id]
                word_len = word_len + get_results(df_part, mode)
        else:
            word_len = df.adjusted_onset - df.adjusted_offset.shift(1)
            word_len = word_len.dropna()
    else:
        raise Exception("Invalid Mode")

    if df.conversation_id.nunique() == 1:
        word_len = [word / 512 for word in word_len]

    return word_len


def plot_results(df, pdf, mode):
    word_len = get_results(df, mode)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(word_len, log=True, bins=100)
    ax.set(
        xlabel="Seconds",
        ylabel="Frequency",
        title=f"All Datum ({len(word_len)} words)",
    )
    pdf.savefig(fig)
    plt.close()

    return pdf


def plot_results_convo(df, pdf, mode, convo_id):
    word_len = get_results(df, mode)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(word_len, log=True, bins=100)
    ax.set(
        xlabel="Seconds",
        ylabel="Frequency",
        title=f"Conversation {convo_id} ({len(word_len)} words)",
    )
    pdf.savefig(fig)
    plt.close()

    return pdf


def plot_results_all_sid(sid_list, pdf, mode):
    fig_all, ax_all = plt.subplots(figsize=(15, 6))

    for sid in sid_list:
        ds = load_pickle(
            os.path.join(
                "data/tfs", str(sid), "pickles", str(sid) + "_full_labels.pkl"
            )
        )
        df = pd.DataFrame(ds["labels"])
        word_len = get_results(df, mode)

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.hist(word_len, log=True, density=True, bins=100)
        ax_all.hist(
            word_len, log=True, density=True, bins=100, alpha=0.5, label=sid
        )
        ax.set(
            xlabel="Seconds",
            ylabel="Density",
            title=f"{sid} All Datum ({len(word_len)} words)",
        )
        pdf.savefig(fig)
    ax_all.set(xlabel="Seconds", ylabel="Density", title="All patients")
    ax_all.legend(loc="upper right")
    pdf.savefig(fig_all)

    return pdf


def add_word_info(df, mode, level="convo"):  # add info for the correct level
    df = df.sort_values(by=["adjusted_onset"])
    df["word_gap"] = df.adjusted_onset - df.adjusted_offset.shift(1)
    df["word_len"] = df.adjusted_offset - df.adjusted_onset
    df["word_lengap"] = df.adjusted_onset - df.adjusted_onset.shift(1)

    if mode == "word_gap":
        if level == "utt":
            df = df.loc[df.production == df.production.shift(1)]
            df = df.loc[df.conversation_id == df.conversation_id.shift(1)]
        elif level == "convo":
            df = df.loc[df.conversation_id == df.conversation_id.shift(1)]

    df.word_gap = df.word_gap / 512
    df.word_len = df.word_len / 512
    df.word_lengap = df.word_lengap / 512

    return df


def main():
    # multiple sids => density plots + combined density plots
    # single sid => frequency plots + frequency plots for each convo
    sid = [625, 676, 7170]
    sid = [7170]
    mode = "word_len"  # length of words
    mode = "word_gap"  # gaps between words (exclude convo gaps)

    if len(sid) == 1:
        sid = sid[0]
        pdf = PdfPages("results/figures/" + str(sid) + "_" + mode + ".pdf")
        ds = load_pickle(
            os.path.join(
                "data/tfs", str(sid), "pickles", str(sid) + "_full_labels.pkl"
            )
        )
        df = pd.DataFrame(ds["labels"])
        df = add_word_info(df, mode, "utt")
        breakpoint()

        plot_results(df, pdf, mode)

        for convo_id in df.conversation_id.unique():
            df_part = df.loc[df["conversation_id"] == convo_id]
            plot_results_convo(df_part, pdf, mode, convo_id)

    else:
        pass
        pdf = PdfPages("results/figures/" + mode + ".pdf")
        plot_results_all_sid(sid, pdf, mode)

    pdf.close()
    return


if __name__ == "__main__":
    main()
