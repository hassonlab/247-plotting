import pickle
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

from tfsplt_utils import load_pickle


def run_pca(pca_to, df, col):
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df[col]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df[col] = pca_output.tolist()

    return df


def logistic(df, x, y):
    print(f"Logistic from {x} to {y}")

    print(f"original # {len(df[y].unique())}")
    g = df.groupby(df[y])
    df = g.filter(lambda x: len(x) >= 100)
    df.reset_index(drop=True, inplace=True)
    print(f"new # {len(df[y].unique())}")

    kfolds = 10
    skf = KFold(n_splits=kfolds, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(len(df)))]

    if x in df.columns:  # logistic
        model = make_pipeline(
            StandardScaler(),
            # PCA(50, whiten=True),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    elif x == "uniform":  # control 1
        model = make_pipeline(
            StandardScaler(),
            # PCA(50, whiten=True),
            DummyClassifier(strategy="uniform"),
        )
        x = "de_emb"
    elif x == "strat":  # control 2
        model = make_pipeline(
            StandardScaler(),
            # PCA(50, whiten=True),
            DummyClassifier(strategy="stratified"),
        )
        x = "de_emb"
    elif x == "freq":  # control 2
        model = make_pipeline(
            StandardScaler(),
            # PCA(50, whiten=True),
            DummyClassifier(strategy="most_frequent"),
        )
        x = "de_emb"

    scores = []
    for i in range(kfolds):
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]
        test_index = folds[test_fold]
        train_index = np.concatenate([folds[j] for j in train_folds])

        X_train = df.loc[train_index, x]
        X_test = df.loc[test_index, x]
        Y_train = df.loc[train_index, y]
        Y_test = df.loc[test_index, y]

        X_train = np.array(X_train.tolist())
        X_test = np.array(X_test.tolist())
        Y_train = np.array(Y_train.tolist())
        Y_test = np.array(Y_test.tolist())

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        df.loc[test_index, "pred"] = preds

        # scores.append(model.score(X_test, Y_test))
        scores.append(balanced_accuracy_score(Y_test, preds))

    # score = sum(df.pred == df[y]) / len(df)
    score = balanced_accuracy_score(df[y], df.pred)
    scores.append(score)
    print(f"Prediction Accuracy: {score}")

    return scores


def main():
    # Loading tf
    layer = ""
    emb_type = "1st"
    emb_type = "all"
    emb_type = "ave"
    # dirname = "results/20230607-whisper-tsne/"
    dirname = "results/20230612-whisper-tsne-no-filter/"
    # dirname = "results/20230613-whisper-medium-podcast/"
    tsne_file = f"all4-whisper-embs-{emb_type}{layer}.pkl"
    # tsne_file = f"777-whisper-embs-{emb_type}{layer}.pkl"
    tsne_file = os.path.join(dirname, tsne_file)
    df = load_pickle(tsne_file)

    # df = df[df.word_freq_overall >= 10]

    plot_dict = {
        "pho": "phoneme",
        "place_artic": "place_of_articulation",
        "manner_artic": "manner_of_articulation",
        "part_of_speech": "part_of_speech",
        # "voice": "voice_or_voiceless",
        # "function_content": "function_or_content",
    }
    embs = [
        "en_emb",
        "de_emb",
        "freq",
        "uniform",
        "strat",
    ]
    dims = [50]

    for dim in dims:  # loop over dim
        results_df = pd.DataFrame(columns={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        print(dim)
        df = run_pca(dim, df, "en_emb")
        df = run_pca(dim, df, "de_emb")
        for plot in plot_dict.keys():
            for emb in embs:
                scores = logistic(df, emb, plot)
                name = f"{emb}-{plot}"
                results_df.loc[name, :] = scores
        results_df.to_csv(
            os.path.join(
                dirname,
                # f"classifier_pca{dim}_filter-sep_{emb_type}_L{layer}.csv",
                f"classifier_pca{dim}_filter-100_{emb_type}_L{layer}-balanced.csv",
            )
        )

    return


if __name__ == "__main__":
    main()
