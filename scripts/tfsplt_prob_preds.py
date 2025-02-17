import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tfsplt_utils import load_pickle


def load_datums_hack(emb, emb_folder):
    emb_df = pd.DataFrame()
    for sid in ["625", "676", "7170", "798"]:
        emb_df_sid = load_pickle(
            f"data/plotting/paper-prob-improb/{emb_folder}/{sid}_{emb}.pkl"
        )
        emb_df_sid["sid"] = sid
        emb_df = pd.concat((emb_df, emb_df_sid))
    emb_df.dropna(subset="true_pred_prob", inplace=True)
    emb_df.reset_index(drop=True, inplace=True)

    emb_df.to_pickle(f"data/plotting/paper-prob-improb/{emb_folder}/all_{emb}.pkl")

    return


def align():
    return
    # datum_top_aligned = datum_top[datum_top.word.isin(datum_bot.word.unique())]
    # datum_bot_aligned = datum_bot[datum_bot.word.isin(datum_top.word.unique())]
    # datum_top.to_pickle(f"{args.sid}_llama3_prob.pkl")
    # datum_bot.to_pickle(f"{args.sid}_llama3_improb.pkl")
    # datum_top_aligned.to_pickle(f"{args.sid}_llama3_prob_a.pkl")
    # datum_bot_aligned.to_pickle(f"{args.sid}_llama3_improb_a.pkl")
    # breakpoint()


def prob_improb(df):
    top = df.true_pred_prob.quantile(70 / 100)
    bot = df.true_pred_prob.quantile(30 / 100)
    df_top = df[df.true_pred_prob >= top]
    df_bot = df[df.true_pred_prob <= bot]

    return df_top, df_bot


def main():

    # gpt2_32 = load_pickle("data/plotting/paper-prob-improb/preds/all_gpt2_32.pkl")
    # gpt2_1024 = load_pickle("data/plotting/paper-prob-improb/preds/all_gpt2_1024.pkl")
    # llama2_32 = load_pickle("data/plotting/paper-prob-improb/preds/all_llama2_32.pkl")
    # llama3_32 = load_pickle("data/plotting/paper-prob-improb/preds/all_llama3_32.pkl")
    # llama3_8192 = load_pickle(
    #     "data/plotting/paper-prob-improb/preds/all_llama3_8192.pkl"
    # )

    # gpt2_32_top, gpt2_32_bot = prob_improb(gpt2_32)
    # gpt2_1024_top, gpt2_1024_bot = prob_improb(gpt2_1024)
    # llama2_32_top, llama2_32_bot = prob_improb(llama2_32)
    # llama3_32_top, llama3_32_bot = prob_improb(llama3_32)
    # llama3_8192_top, llama3_8192_bot = prob_improb(llama3_8192)

    # gpt2_32_top.drop_duplicates(subset=["word", "part_of_speech"], inplace=True)

    # load_datums_hack("gpt2_32_prob", "prob-improb")
    # load_datums_hack("gpt2_32_improb", "prob-improb")
    # load_datums_hack("gpt2_1024_prob", "prob-improb")
    # load_datums_hack("gpt2_1024_improb", "prob-improb")
    # load_datums_hack("llama2_32_prob", "prob-improb")
    # load_datums_hack("llama2_32_improb", "prob-improb")
    # load_datums_hack("llama3_32_prob", "prob-improb")
    # load_datums_hack("llama3_32_improb", "prob-improb")
    # load_datums_hack("llama3_8192_prob", "prob-improb")
    # load_datums_hack("llama3_8192_improb", "prob-improb")
    # load_datums_hack("gpt2_32_prob_a", "prob-improb")
    # load_datums_hack("gpt2_32_improb_a", "prob-improb")
    # load_datums_hack("gpt2_1024_prob_a", "prob-improb")
    # load_datums_hack("gpt2_1024_improb_a", "prob-improb")
    # load_datums_hack("llama2_32_prob_a", "prob-improb")
    # load_datums_hack("llama2_32_improb_a", "prob-improb")
    # load_datums_hack("llama3_32_prob_a", "prob-improb")
    # load_datums_hack("llama3_32_improb_a", "prob-improb")
    # load_datums_hack("llama3_8192_prob_a", "prob-improb")
    # load_datums_hack("llama3_8192_improb_a", "prob-improb")
    # load_datums_hack("llama2_32_prob", "prob-improb2")
    # load_datums_hack("llama2_32_mid", "prob-improb2")
    # load_datums_hack("llama2_32_improb", "prob-improb2")
    # load_datums_hack("gpt2_32", "preds2")
    load_datums_hack("llama2_32_cor", "prob-improb2")
    load_datums_hack("llama2_32_incor", "prob-improb2")
    load_datums_hack("llama2_32_cor_a", "prob-improb2")
    load_datums_hack("llama2_32_incor_a", "prob-improb2")
    breakpoint()

    # prob_str = "prob"
    # improb_str = "improb"

    # gpt2_prob = load_pickle(
    #     f"data/plotting/paper-prob-improb/hack1-datums/all_gpt2_{prob_str}.pkl"
    # )
    # llama3_prob = load_pickle(
    #     f"data/plotting/paper-prob-improb/hack1-datums/all_llama3_{prob_str}.pkl"
    # )
    # gpt2_improb = load_pickle(
    #     f"data/plotting/paper-prob-improb/hack1-datums/all_gpt2_{improb_str}.pkl"
    # )
    # llama3_improb = load_pickle(
    #     f"data/plotting/paper-prob-improb/hack1-datums/all_llama3_{improb_str}.pkl"
    # )

    # gpt2_all = load_pickle(f"data/plotting/paper-prob-improb/hack2-datums/all_gpt2.pkl")
    # llama3_all = load_pickle(
    #     f"data/plotting/paper-prob-improb/hack2-datums/all_llama3.pkl"
    # )

    # breakpoint()

    # gpt2_prob.groupby("part_of_speech").size().sort_values()
    # gpt2_prob.drop_duplicates(subset=["word", "part_of_speech"]).groupby(
    #     "part_of_speech"
    # ).size().sort_values()
    # new_df = gpt2_improb[~gpt2_improb.word.isin(llama3_improb.word.unique())]
    # new_df = llama3_improb[~llama3_improb.word.isin(gpt2_improb.word.unique())]
    # new_df.drop_duplicates(subset=["word", "part_of_speech"], inplace=True)
    # breakpoint()

    # preds = pd.DataFrame(
    #     {"gpt2": gpt2_prob.true_pred_prob_x, "llama3": llama3_prob.true_pred_prob_y}
    # )

    # Only take token is root (FOR HACK 2)
    # gpt2_df = gpt2_df.loc[
    #     gpt2_df["gpt2-xl_token_is_root"], ("word", "adjusted_onset", "true_pred_prob")
    # ]
    # llama3_df = llama3_df.loc[
    #     llama3_df["Meta-Llama-3-8B_token_is_root"],
    #     ("word", "adjusted_onset", "true_pred_prob"),
    # ]
    # preds = gpt2_df.merge(llama3_df, how="inner", on=["word", "adjusted_onset"])
    # preds = pd.DataFrame(
    #     {"gpt2": preds.true_pred_prob_x, "llama3": preds.true_pred_prob_y}
    # )

    # fig, ax = plt.subplots(figsize=(50, 50))
    # sns.scatterplot(data=preds, x="gpt2", y="llama3", size=2)
    # ax.set_xlabel("gpt2 preds", fontsize=100)
    # ax.set_ylabel("llama3 preds", fontsize=100)
    # ax.tick_params(axis="both", which="major", labelsize=80)
    # ax.tick_params(axis="both", which="minor", labelsize=80)
    # plt.savefig("scatter.jpeg")

    return


if __name__ == "__main__":
    main()
