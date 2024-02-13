import pickle
import os
import argparse
import sys

import numpy as np
import pandas as pd
import nltk

from tfsplt_utils import load_pickle


def ave_emb(datum):
    print("Averaging embeddings across tokens")

    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "embeddings"
    ].apply(lambda x: mean_emb(x))
    mean_embs = pd.DataFrame(mean_embs)

    # replace embeddings
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]
    mean_embs.set_index(datum.index, inplace=True)

    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings"] = mean_embs["embeddings"]
    datum = datum2  # reassign back to datum

    return datum


def aggregate_df():
    print("Aggregating Data")

    subjects = [625, 676, 7170, 798]

    whisper_en = f"pickles/embeddings/symbolic-lang/full/cnxt_0001/layer_00.pkl"

    # load in data
    whisper_df = pd.DataFrame()

    for subj in subjects:
        temp_en_df = load_pickle(f"data/pickling/tfs/{subj}/{whisper_en}")
        whisper_df = pd.concat([whisper_df, temp_en_df])

    breakpoint()
    prod_df = whisper_df[whisper_df.production == 1]
    comp_df = whisper_df[whisper_df.production == 0]
    prod_df.groupby(prod_df.part_of_speech).size().sort_values(ascending=False)
    # whisper_df.groupby(whisper_df.part_of_speech).size().sort_values(ascending=False)/len(whisper_df)
    # whisper_df = ave_emb(whisper_df)

    return whisper_df


def main():
    df = aggregate_df()

    return


if __name__ == "__main__":
    main()
