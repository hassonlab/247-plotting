import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def main():
    layers = np.arange(0, 25)

    data = []
    data_de = []
    data_en = []
    for layer in layers:
        en_pkl = f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-medium.en-encoder-new/full/cnxt_0001/layer_{layer:02d}.pkl"
        de_pkl = f"/scratch/gpfs/kw1166/247-pickling/results/podcast/777/pickles/embeddings/whisper-medium.en-decoder/full/cnxt_0001/layer_{layer:02d}.pkl"

        df_en = load_pickle(en_pkl)
        df = load_pickle(de_pkl)
        print(f"Original datum len: {len(df)}")
        df_en = run_pca(50, df_en)
        df = run_pca(50, df)

        # def calc_pdist(x):
        #     emb = np.reshape(x, (12, 1024))
        #     pdists = pdist(emb, metric="correlation")
        #     return pdists.mean()
        # data.append(df.embeddings.apply(calc_pdist))

        embs = df.embeddings.tolist()
        embs = np.array(embs)
        print(embs.shape)
        sm2 = pdist(embs, metric="correlation")
        breakpoint()
        data_de.append(sm2.mean())

        embs = df_en.embeddings.tolist()
        embs = np.array(embs)
        print(embs.shape)
        sm2 = pdist(embs, metric="correlation")
        data_en.append(sm2.mean())

    breakpoint()
    data = pd.concat(data, axis=1)
    data.columns = layers

    return


if __name__ == "__main__":
    main()
