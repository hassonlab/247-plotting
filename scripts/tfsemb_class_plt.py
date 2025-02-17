import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

# plt.style.use("data/plotting/paper-prob-improb/paper.mlpstyle")
plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")


# get classifier bar plot results
def get_classifier_results(filename, class_type, class_type_plt, class_cat):
    df = pd.read_csv(filename)

    # whole = df.loc[:, "10"].tolist()
    df.drop(columns=["10"], inplace=True)
    means = df.mean(axis=1).tolist()
    stds = df.std(axis=1).tolist()

    results_df = pd.DataFrame(
        {
            "balanced accuracy": means,
            "std": stds,
            "class_type": class_type,
            "class_cat": class_cat,
        }
    )

    results_df = results_df[results_df.class_type.isin(class_type_plt)]

    return results_df


# plot classifier bar plots
def plot_classifier_bar(results_df, filename, colors):
    dfp = results_df.pivot(
        index="class_cat", columns="class_type", values="balanced accuracy"
    )
    yerr = results_df.pivot(index="class_cat", columns="class_type", values="std")
    dfp.plot(
        kind="bar",
        yerr=yerr,
        rot=0,
        color=colors,
        error_kw=dict(ecolor="black", elinewidth=1, capsize=1),
    )

    plt.savefig(filename)
    plt.close()

    return


def box_plot(class_results, class_type, class_type_plt, class_cat):
    df = pd.read_csv(class_results)
    df.drop(columns=["10"], inplace=True)
    df["class_type"] = class_type
    df["class_cat"] = class_cat
    df = df[df.class_type.isin(class_type_plt)]
    breakpoint()

    df = df.melt(
        id_vars=["class_type", "class_cat"],
        value_vars=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )

    colors = ["red", "blue", "gray"]
    colors2 = ["black"]
    # fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig, axes = plt.subplots(1, 1)
    sns.boxplot(
        data=df,
        x="class_cat",
        y="value",
        hue="class_type",
        ax=axes,
        # order=MODELS,
        palette=colors,
        showfliers=False,
    )
    sns.stripplot(
        data=df,
        x="class_cat",
        y="value",
        hue="class_type",
        ax=axes,
        # order=MODELS,
        size=2,
        palette=colors2,
        dodge=True,
        # color=".3",
    )
    plt.savefig("bar2.svg")
    return


def box_plot_layers(class_results, class_type, class_type_plt, class_cat):
    layers = [0, 1, 2, 3, 4]
    df_all = pd.DataFrame()
    for layer in layers:
        df = pd.read_csv(class_results % layer)
        df.drop(columns=["10"], inplace=True)
        df["class_type"] = class_type
        df["class_cat"] = class_cat
        df = df[df.class_type.isin(class_type_plt)]

        df = df.melt(
            id_vars=["class_type", "class_cat"],
            value_vars=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )
        df["layer"] = layer
        df_all = pd.concat((df_all, df))

    df_all = df_all[~(df_all.class_type.eq("3control") & df_all.layer.ge(1))]
    df_all.loc[df_all.class_type.eq("3control"), "layer"] = 5
    df_lang = df_all[~df_all.class_type.eq("1speech")]
    df_speech = df_all[~df_all.class_type.eq("2language")]

    lang_colors = [
        "paleturquoise",
        "darkturquoise",
        "dodgerblue",
        "blue",
        "darkblue",
        "grey",
    ]
    speech_colors = [
        "mistyrose",
        "lightcoral",
        "indianred",
        "tomato",
        "red",
        "grey",
    ]
    colors2 = ["black"]
    # fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig, axes = plt.subplots(1, 1)
    sns.boxplot(
        data=df_lang,
        x="class_cat",
        y="value",
        hue="layer",
        ax=axes,
        # order=MODELS,
        palette=lang_colors,
        showfliers=False,
    )
    sns.stripplot(
        data=df_lang,
        x="class_cat",
        y="value",
        hue="layer",
        ax=axes,
        # order=MODELS,
        size=2,
        palette=colors2,
        dodge=True,
        # color=".3",
    )
    plt.savefig("bar-lang.svg")
    fig, axes = plt.subplots(1, 1)
    sns.boxplot(
        data=df_speech,
        x="class_cat",
        y="value",
        hue="layer",
        ax=axes,
        # order=MODELS,
        palette=speech_colors,
        showfliers=False,
    )
    sns.stripplot(
        data=df_speech,
        x="class_cat",
        y="value",
        hue="layer",
        ax=axes,
        # order=MODELS,
        size=2,
        palette=colors2,
        dodge=True,
        # color=".3",
    )
    plt.savefig("bar-speech.svg")
    return


def main():
    # Get dataframe for classification results
    class_type = [  # types of classifier (do not change)
        "1speech",
        "2language",
        "3control",
        "uniform",
        "strat",
    ] * 4
    class_type_plt = ["1speech", "2language", "3control"]  # types to plot (can change)
    class_cat = np.repeat(
        ["1Phoneme", "2PoA", "3MoA", "4PoS"], len(class_type) / 4
    )  # categories

    # box_plot(
    #     "results/20230612-whisper-tsne-no-filter/classifier_pca50_filter-100_ave_L.csv",
    #     class_type,
    #     class_type_plt,
    #     class_cat,
    # )
    box_plot_layers(
        "results/20230612-whisper-tsne-no-filter/classifier_pca50_filter-100_ave_L%s.csv",
        class_type,
        class_type_plt,
        class_cat,
    )

    # class_results_df = get_classifier_results(
    #     "results/20230612-whisper-tsne-no-filter/classifier_pca50_filter-100_ave_L.csv",
    #     class_type,
    #     class_type_plt,
    #     class_cat,
    # )

    # # Plot classifier bar plots
    # class_bar_colors = ["red", "blue", "grey"]
    # class_bar_name = "results/barplot.jpeg"
    # plot_classifier_bar(class_results_df, class_bar_name, class_bar_colors)
    return


if __name__ == "__main__":
    main()
