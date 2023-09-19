import os
import string

import numpy as np
import pandas as pd
import pickle


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
    df = pd.DataFrame.from_dict(datum)
    return df


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    is_nan = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~is_nan]

    return df


def load_glove_embeddings(sid):
    pickle_dir = f"data/pickling/tfs/{sid}/pickles/embeddings/glove50/full"
    glove_base_df_path = os.path.join(pickle_dir, "base_df.pkl")
    glove_emb_df_path = os.path.join(
        pickle_dir,
        "cnxt_0001",
        "layer_01.pkl",
    )

    glove_base_df = load_datum(glove_base_df_path)
    glove_emb_df = load_datum(glove_emb_df_path)
    if len(glove_base_df) != len(glove_emb_df):
        glove_df = pd.merge(
            glove_base_df, glove_emb_df, left_index=True, right_index=True
        )
    else:
        glove_base_df.reset_index(drop=False, inplace=True)
        glove_df = pd.concat([glove_base_df, glove_emb_df], axis=1)
    glove_df = glove_df[glove_df["in_gpt2-xl"]]
    glove_df = glove_df.loc[:, ["adjusted_onset", "word", "embeddings"]]

    return glove_df


def process_embeddings(df, sid):
    """Process the datum embeddings based on input arguments

    Args:
        args (namespace): commandline arguments
        df : raw datum as a DataFrame

    Returns:
        DataFrame: processed datum with correct embeddings
    """

    # drop NaN / None embeddings
    df = drop_nan_embeddings(df)
    df = remove_punctuation(df)

    # add prediction embeddings (force to glove)
    mask = df["in_glove50"] & df["gpt2-xl_token_is_root"]
    df = df[mask]
    df.drop(
        ["embeddings"],
        axis=1,
        errors="ignore",
        inplace=True,
    )  # delete current embeddings
    glove_df = load_glove_embeddings(sid)
    df = df[df.adjusted_onset.notna()]
    glove_df = glove_df[glove_df.adjusted_onset.notna()]
    df = df.merge(glove_df, how="inner", on=["adjusted_onset", "word"])

    return df


def filter_datum(df):
    """Process/clean/filter datum based on args

    Args:
        args (namespace): commandline arguments
        df: processed datum
        stitch: stitch index

    Returns:
        DataFrame: filtered datum
    """

    ## Trimming datum
    # df = trim_datum(args, df)  # trim edges

    # create mask for further filtering
    common = np.repeat(True, len(df))

    # get rid of tokens without onset/offset
    common &= df.adjusted_onset.notna()
    common &= df.adjusted_offset.notna()
    common &= df.onset.notna()
    common &= df.offset.notna()

    # get word gap
    df["word_gap"] = df.adjusted_onset - df.adjusted_offset.shift()
    df["word_len"] = df.adjusted_offset - df.adjusted_onset

    # get rid of tokens without proper speaker
    speaker_mask = df.speaker.str.contains("Speaker")
    common &= speaker_mask

    # filter based on arguments: nonwords, word_freq
    common &= ~df.is_nonword

    df = df[common]

    return df


def read_datum(sid):
    """Load, process, and filter datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        DataFrame: processed and filtered datum
    """
    emb_df = load_datum(
        f"data/pickling/tfs/{sid}/pickles/embeddings/gpt2-xl/full/cnxt_1024/layer_48.pkl"
    )
    base_df = load_datum(
        f"data/pickling/tfs/{sid}/pickles/embeddings/gpt2-xl/full/base_df.pkl"
    )

    base_df.reset_index(drop=False, inplace=True)
    df = pd.concat([base_df, emb_df], axis=1)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_embeddings(df, sid)
    print(f"After processing: Datum now has {len(df)} words")

    df = filter_datum(df)
    print(f"After filtering: Datum now has {len(df)} words")

    percentile = 30
    top = df.true_pred_prob.quantile(1 - percentile / 100)
    bot = df.true_pred_prob.quantile(percentile / 100)
    mid_low = df.true_pred_prob.quantile(35 / 100)
    mid_high = df.true_pred_prob.quantile(65 / 100)

    df = df[  # only select second word onwards for each utt (for word gap)
        (df.production.shift() == df.production)
        & (df.conversation_id.shift() == df.conversation_id)
    ]
    df["sid"] = sid
    df = df.loc[
        :,
        (
            "sid",
            "word",
            "onset",
            "offset",
            "adjusted_onset",
            "adjusted_offset",
            "word_freq_overall",
            "production",
            "speaker",
            "true_pred_prob",
            "true_pred_rank",
            "surprise",
            "entropy",
            "word_gap",
            "word_len",
        ),
    ]

    df_top = df[df.true_pred_prob >= top].copy()
    df_bot = df[df.true_pred_prob <= bot].copy()
    df_mid = df[(df.true_pred_prob >= mid_low) & (df.true_pred_prob <= mid_high)].copy()

    df_top_aligned = df_top[df_top.word.isin(df_bot.word.unique())]
    df_bot_aligned = df_bot[df_bot.word.isin(df_top.word.unique())]

    df_top["word_num"] = (
        df_top.sort_values(["word", "true_pred_prob"], ascending=False)
        .groupby("word")
        .cumcount()
        + 1
    )
    df_bot["word_num"] = (
        df_bot.sort_values(["word", "true_pred_prob"], ascending=True)
        .groupby("word")
        .cumcount()
        + 1
    )
    df_top_numaligned = df_top.merge(
        df_bot.loc[:, ("word", "word_num")],
        how="inner",
        on=["word", "word_num"],
    )
    df_bot_numaligned = df_bot.merge(
        df_top.loc[:, ("word", "word_num")],
        how="inner",
        on=["word", "word_num"],
    )

    return (
        df,
        df_top,
        df_bot,
        df_mid,
        # df_top_aligned,
        # df_bot_aligned,
        # df_top_numaligned,
        # df_bot_numaligned,
    )


def read_datum2(sid):
    """Load, process, and filter datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        DataFrame: processed and filtered datum
    """
    pred_df = load_datum(
        f"data/pickling/tfs/{sid}/pickles/embeddings/gpt2-xl/full/cnxt_1004/layer_00.pkl"
    )
    base_df = load_datum(
        f"data/pickling/tfs/{sid}/pickles/embeddings/gpt2-xl/full/base_df.pkl"
    )

    base_df.reset_index(drop=False, inplace=True)
    df = pd.concat([base_df, pred_df], axis=1)
    print(f"After loading: Datum loads with {len(df)} words")

    # drop NaN / None embeddings
    df = df[~df.top1_pred.isna()]
    df = remove_punctuation(df)
    print(f"After processing: Datum now has {len(df)} words")

    df = filter_datum(df)
    print(f"After filtering: Datum now has {len(df)} words")

    percentile = 30
    df["true_pred_prob0"] = df.true_pred_prob.apply(lambda x: (x[0]))
    top = df.true_pred_prob0.quantile(1 - percentile / 100)
    bot = df.true_pred_prob0.quantile(percentile / 100)
    mid_low = df.true_pred_prob0.quantile(35 / 100)
    mid_high = df.true_pred_prob0.quantile(65 / 100)

    df = df[  # only select second word onwards for each utt (for word gap)
        (df.production.shift() == df.production)
        & (df.conversation_id.shift() == df.conversation_id)
    ]
    df["sid"] = sid
    df = df.loc[
        :,
        (
            "sid",
            "word",
            "onset",
            "offset",
            "adjusted_onset",
            "adjusted_offset",
            "word_freq_overall",
            "production",
            "speaker",
            "top1_pred",
            "top1_pred_prob",
            "true_pred_prob",
            "true_pred_prob0",
            "true_pred_rank",
        ),
    ]

    df_top = df[df.true_pred_prob0 >= top].copy()
    df_bot = df[df.true_pred_prob0 <= bot].copy()
    df_mid = df[
        (df.true_pred_prob0 >= mid_low) & (df.true_pred_prob0 <= mid_high)
    ].copy()

    return (
        df,
        df_top,
        df_bot,
        df_mid,
    )


def main():
    df = pd.DataFrame()
    df_top = pd.DataFrame()
    df_bot = pd.DataFrame()
    df_top_aligned = pd.DataFrame()
    df_bot_aligned = pd.DataFrame()
    df_top_numaligned = pd.DataFrame()
    df_bot_numaligned = pd.DataFrame()
    df_mid = pd.DataFrame()

    for sid in [625, 676, 7170, 798]:
        # df_topa_sid, df_bota_sid, df_topna_sid, df_botna_sid = read_datum(sid)
        # df_top_aligned = pd.concat((df_top_aligned, df_topa_sid))
        # df_bot_aligned = pd.concat((df_bot_aligned, df_bota_sid))
        # df_top_numaligned = pd.concat((df_top_numaligned, df_topna_sid))
        # df_bot_numaligned = pd.concat((df_bot_numaligned, df_botna_sid))

        df_sid, df_top_sid, df_bot_sid, df_mid_sid = read_datum2(sid)
        df = pd.concat((df, df_sid))
        df_top = pd.concat((df_top, df_top_sid))
        df_bot = pd.concat((df_bot, df_bot_sid))
        df_mid = pd.concat((df_mid, df_mid_sid))

    pickle_dir = "data/plotting/paper-prob-improb/datums/"
    df.to_pickle(os.path.join(pickle_dir, "df_pred.pkl"))
    df_top.to_pickle(os.path.join(pickle_dir, "df_pred_top.pkl"))
    df_bot.to_pickle(os.path.join(pickle_dir, "df_pred_bot.pkl"))
    df_mid.to_pickle(os.path.join(pickle_dir, "df_pred_mid.pkl"))
    # df_top_aligned.to_pickle(os.path.join(pickle_dir, "df_top_a.pkl"))
    # df_bot_aligned.to_pickle(os.path.join(pickle_dir, "df_bot_a.pkl"))
    # df_top_numaligned.to_pickle(os.path.join(pickle_dir, "df_top_na.pkl"))
    # df_bot_numaligned.to_pickle(os.path.join(pickle_dir, "df_bot_na.pkl"))

    return


if __name__ == "__main__":
    main()
