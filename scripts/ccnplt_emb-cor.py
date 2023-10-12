import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from tfsplt_utils import load_pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def run_pca(pca_to, df):
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


def window_cors(proj, model, layers):
    """Compute correlations between encoder windows

    Args:
        layers ([int]): list of layer nums

    Returns:
    """
    # data = []
    for layer in layers:
        EN_PATH = {
            (
                "podcast",
                "medium",
            ): f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-medium.en-encoder-new/full/cnxt_0001/layer_{layer:02d}.pkl",
            (
                "podcast",
                "tiny",
            ): f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-tiny.en-encoder-leo/full/cnxt_0001/layer_{layer:02d}.pkl",
        }
        EMB_SIZE = {"medium": 1024, "tiny": 384}
        WIN_SIZE = {"podcast": 12, "tfs": 10}

        filepath = []
        filepath2 = []
        for sid in [625, 676, 7170, 798]:
            filepath.append(
                f"/scratch/gpfs/kw1166/247-pickling/results/tfs/{sid}/pickles/embeddings/whisper-medium.en-encoder/full/cnxt_0001/layer_{layer:02d}.pkl"
            )
            filepath2.append(
                f"/scratch/gpfs/kw1166/247-pickling/results/tfs/{sid}/pickles/embeddings/whisper-tiny.en-encoder/full/cnxt_0001/layer_{layer:02d}.pkl"
            )
        EN_PATH[("tfs", "medium")] = filepath
        EN_PATH[("tfs", "tiny")] = filepath2

        if proj == "podcast":
            en_pkl = EN_PATH[(proj, model)]
            df_en = load_pickle(en_pkl)
        elif proj == "tfs":
            df_en = pd.DataFrame([])
            for pkl_path in EN_PATH[(proj, model)]:
                df_en_sid = load_pickle(pkl_path)
                df_en = pd.concat((df_en, df_en_sid))

        print(f"Original datum len: {len(df_en)}")

        def calc_pdist(x):
            emb = np.reshape(x, (WIN_SIZE[proj], EMB_SIZE[model]))
            pdists = pdist(emb, metric="correlation")
            return pdists

        mean_cors = df_en.embeddings.apply(calc_pdist).mean()
        square_mean_cors = squareform(mean_cors)
        square_mean_cors = 1 - square_mean_cors

        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            square_mean_cors,
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            # cbar_kws={"shrink": 0.5},
        )
        plt.savefig(f"{layer}.png")
        plt.close()

        # data.append(mean_cors)

    breakpoint()
    # data = pd.concat(data, axis=1)

    return


def old_main(layers):
    data = []
    data_de = []
    data_en = []
    for layer in layers:
        en_pkl = f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-medium.en-encoder-new/full/cnxt_0001/layer_{layer:02d}.pkl"
        de_pkl = f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-medium.en-decoder/full/cnxt_0001/layer_{layer:02d}.pkl"

        df_en = load_pickle(en_pkl)
        df_de = load_pickle(de_pkl)
        print(f"Original datum len: {len(df_en)} {len(df_de)}")
        df_en = run_pca(50, df_en)
        df_de = run_pca(50, df_de)

        def calc_pdist(x):
            emb = np.reshape(x, (12, 1024))
            pdists = pdist(emb, metric="correlation")
            return pdists.mean()

        breakpoint()

        data.append(df_en.embeddings.apply(calc_pdist))

        # embs = df.embeddings.tolist()
        # embs = np.array(embs)
        # print(embs.shape)
        # sm2 = pdist(embs, metric="correlation")
        # breakpoint()
        # data_de.append(sm2.mean())

        # embs = df_en.embeddings.tolist()
        # embs = np.array(embs)
        # print(embs.shape)
        # sm2 = pdist(embs, metric="correlation")
        # data_en.append(sm2.mean())

    breakpoint()
    data = pd.concat(data, axis=1)
    data.columns = layers

    return


def main():
    layers = np.arange(0, 25)
    layers = np.arange(0, 5)
    layers = np.arange(5, 25)

    # window_cors("podcast", "medium", layers)
    window_cors("tfs", "medium", layers)

    return


if __name__ == "__main__":
    main()
