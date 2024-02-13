import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum


def load_datum(file_name):
    """Read raw datum

    Args:
        filename: raw datum full file path

    Returns:
        DataFrame: datum
    """
    datum = load_pickle(file_name)
    df = pd.DataFrame(datum["labels"])
    return df


def main():
    df = pd.DataFrame([])
    for sid in [625, 676, 7170, 798]:
        pkl = f"data/pickling/tfs/{sid}/pickles/{sid}_full_labels.pkl"
        df_sid = load_datum(pkl)
        df = pd.concat((df, df_sid))

    df["word_len"] = (df.offset - df.onset) * 1000 / 512
    df_short = df[(df["word_len"] > 0) & (df["word_len"] <= 1000)].copy()
    df["word_str_len"] = df.word.str.len()
    fig, axes = plt.subplots(figsize=(20, 10))
    axes.hist(df["word_str_len"], bins="auto")
    plt.yscale("log")
    plt.savefig("word_len_freq-log.png")
    breakpoint()

    # 132 negative, 11 bigger than 2s, 171 bigger than 1s

    # df.word_len.describe()
    # count    510512.000000
    # mean        116.474004
    # std          86.526358
    # min       -2894.692400
    # 25%          61.440000
    # 50%          97.000000
    # 75%         148.480000
    # max        9635.584000

    # df_short.word_len.describe()
    # count    510209.000000
    # mean        116.192078
    # std          81.032447
    # min           0.000000
    # 25%          61.440000
    # 50%          97.000000
    # 75%         148.480000
    # max         998.40000

    fig, axes = plt.subplots(figsize=(20, 10))
    axes.hist(df["word_len"], bins="auto")
    plt.yscale("log")
    plt.savefig("word_len_freq-log.png")

    fig, axes = plt.subplots(figsize=(20, 10))
    axes.hist(df_short["word_len"], bins="auto")
    plt.yscale("log")
    plt.savefig("word_len_short_freq-log.png")

    df_short["win_num"] = (df_short["word_len"] - 52.5) // 20 + 2
    df_short.loc[df_short["win_num"] < 1, "win_num"] = 1

    # df_short.win_num.describe()
    # count    510209.000000
    # mean          4.731955
    # std           4.015259
    # min           1.000000
    # 25%           2.000000
    # 50%           4.000000
    # 75%           6.000000
    # max          49.000000

    # fig, axes = plt.subplots(figsize=(20, 10))
    # axes.hist(df_short["win_num"], bins=50, density=True)
    # plt.savefig("win_num_density.png")

    # fig, axes = plt.subplots(figsize=(20, 10))
    # axes.hist(df_short["win_num"], bins=50, density=True)
    # plt.yscale("log")
    # plt.savefig("win_num_freq-log.png")

    # fig, axes = plt.subplots(figsize=(20, 10))
    # axes.hist(df_short["win_num"], bins=50, density=True, cumulative=True)
    # plt.savefig("win_num_ecdf.png")

    plt.close()

    breakpoint()

    return


if __name__ == "__main__":
    main()
