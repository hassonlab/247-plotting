import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from tfsplt_utils import load_pickle

import gensim.downloader as api


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def tsne(whisper_df, col):
    print(f"Doing t-SNE on {col}")
    tsne = TSNE(n_components=2, perplexity=50, random_state=329)
    embs = pd.DataFrame(np.vstack(whisper_df[col]))
    projections = pd.DataFrame(tsne.fit_transform(embs))
    return projections


def do_tsne(df, col, save_name):
    tsne_results = tsne(df, col)
    df = df.reset_index(drop=True)
    df = df.assign(
        x=tsne_results[0],
        y=tsne_results[1],
    )
    df.to_pickle(f"data/plotting/paper-prob-improb/datums/{save_name}.pkl")

    return df


def ave_emb_wordlevel(datum):
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2["true_pred_prob"] = datum2.groupby("word")["true_pred_prob"].transform(
        np.mean
    )
    datum2["true_pred_rank"] = datum2.groupby("word")["true_pred_rank"].transform(
        np.mean
    )
    datum2["surprise"] = datum2.groupby("word")["surprise"].transform(np.mean)
    datum2["entropy"] = datum2.groupby("word")["entropy"].transform(np.mean)
    datum2.drop_duplicates(["word"], inplace=True, ignore_index=True)

    return datum2


def main():
    print("Running")
    df_top_a = load_pickle("data/plotting/paper-prob-improb/datums/df_top_a.pkl")
    df_bot_a = load_pickle("data/plotting/paper-prob-improb/datums/df_bot_a.pkl")
    df_top_na = load_pickle("data/plotting/paper-prob-improb/datums/df_top_na.pkl")
    df_bot_na = load_pickle("data/plotting/paper-prob-improb/datums/df_bot_na.pkl")
    df_top = load_pickle("data/plotting/paper-prob-improb/datums/df_top.pkl")
    df_bot = load_pickle("data/plotting/paper-prob-improb/datums/df_bot.pkl")

    df_top_a["pred"] = "top-aligned"
    df_bot_a["pred"] = "bot-aligned"
    df_top_na["pred"] = "top-num-aligned"
    df_bot_na["pred"] = "bot-num-aligned"
    df_top["pred"] = "top-original"
    df_bot["pred"] = "bot-original"

    df_top_plot = pd.concat((df_top_na, df_top_a, df_top))
    df_bot_plot = pd.concat((df_bot_na, df_bot_a, df_bot))

    # tsne_top = df_top_plot.drop_duplicates(subset=["word"], keep="first")
    # tsne_bot = df_bot_plot.drop_duplicates(subset=["word"], keep="first")
    # glove = api.load("glove-wiki-gigaword-50")
    # tsne_top.loc[:, "embeddings"] = tsne_top.word.str.lower().apply(
    #     lambda x: get_vector(x, glove)
    # )
    # tsne_bot.loc[:, "embeddings"] = tsne_bot.word.str.lower().apply(
    #     lambda x: get_vector(x, glove)
    # )
    # do_tsne(tsne_top, "embeddings", "tsne-top")
    # do_tsne(tsne_bot, "embeddings", "tsne-bot")

    tsne_top = load_pickle("data/plotting/paper-prob-improb/datums/tsne-top.pkl")
    tsne_bot = load_pickle("data/plotting/paper-prob-improb/datums/tsne-bot.pkl")
    breakpoint()


if __name__ == "__main__":
    main()
