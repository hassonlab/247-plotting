import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def read_results(filename, groups, groups_plt, columns, color):
    dirname = "results/20230607-whisper-tsne/"
    dirname = "results/20230612-whisper-tsne-no-filter/"
    # dirname = "results/20230613-whisper-medium-podcast/"
    df = pd.read_csv(os.path.join(dirname, filename))

    whole = df.loc[:, "10"].tolist()
    df.drop(columns=["10"], inplace=True)
    means = df.mean(axis=1).tolist()
    stds = df.std(axis=1).tolist()

    results_df = pd.DataFrame(
        {
            "balanced accuracy": means,
            "std": stds,
            "groups": groups,
            "columns": columns,
            "colors": color,
        }
    )

    results_df = results_df[results_df.groups.isin(groups_plt)]

    return results_df


groups = [
    "1speech",
    "2language",
    "3control",
    "uniform",
    "strat",
] * 4
groups_plt = ["1speech", "2language", "3control"]
color = ["red", "blue", "grey", "black", "brown"] * 4
columns = np.repeat(["1Phoneme", "2PoA", "3MoA", "4PoS"], len(groups) / 4)


# results_df = pd.DataFrame()
# control = True
# for layer in np.arange(0, 5):
#     temp_df = read_results(
#         f"classifier_pca50_filter-100_ave_L{layer:01}.csv",
#         groups,
#         groups_plt,
#         columns,
#         color,
#     )
#     if control:
#         control = False
#     else:
#         temp_df = temp_df[~temp_df.groups.str.contains("control")]
#     temp_df["groups"] = temp_df.groups + f"{layer:01}"
#     results_df = pd.concat([results_df, temp_df])

# cmap = plt.cm.get_cmap("winter")
# cmap_colors = []
# for i in np.arange(0, 25):
#     cmap_colors.append(cmap(i / 25))
# cmap_colors.append((0, 0, 0, 0.8))
# lang_filter = results_df.groups.str.contains("language")
# speech_filter = results_df.groups.str.contains("speech")
# results_df = results_df[~lang_filter]
# breakpoint()
# col_filter = results_df["columns"].str.contains("Phoneme")
# col_filter2 = results_df["columns"].str.contains("PoS")
# results_df = results_df[col_filter | col_filter2]
# breakpoint()
results_df = read_results(
    "classifier_pca50_filter-100_ave_L-balanced.csv",
    groups,
    groups_plt,
    columns,
    color,
)
breakpoint()

# plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")


# Bar plot with seaborn
# colors = results_df.colors
# barplt = sns.barplot(
#     data=results_df,
#     x="columns",
#     y="balanced accuracy",
#     hue="groups",
#     palette=colors,
# )

# Bar plot with matplotlib (has error bars)
dfp = results_df.pivot(
    index="columns", columns="groups", values="balanced accuracy"
)
yerr = results_df.pivot(index="columns", columns="groups", values="std")
dfp.plot(
    kind="bar",
    yerr=yerr,
    rot=0,
    color=["red", "blue", "grey"],
    # color=cmap_colors,
    # color=[
    #     "paleturquoise",
    #     "darkturquoise",
    #     "dodgerblue",
    #     "blue",
    #     "darkblue",
    #     "mistyrose",
    #     "lightcoral",
    #     "indianred",
    #     "tomato",
    #     "red",
    #     "grey",
    #     "black",
    # ],
    error_kw=dict(ecolor="black", elinewidth=1, capsize=1),
    # legend=None,
    # figsize=(12, 6),
)

plt.savefig("barplot.png")
# plt.savefig("barplot-speech.svg")
plt.close()
