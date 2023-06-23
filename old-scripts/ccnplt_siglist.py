import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from tfsplt_utils import read_folder


def save_file(df, filename):

    df_all = df.loc[:, ("subject", "electrode")]
    df_all.to_csv(f"data/plotting/ccn-sig-file-{filename}.csv", index=False)

    areas = {
        "astg": ["rSTG"],
        "mstg": ["mSTG"],
        # "ifg": ["IFG"],
        # "stg": ["STG"],
        # "sm": ["precentral", "postcentral", "premotor", "postcg"],
    }
    for key, val in areas.items():
        # df_area = df.loc[df.princeton_class.isin(val), ("subject", "electrode")]
        df_area = df.loc[df.NYU_class2.isin(val), ("subject", "electrode")]
        df_area.to_csv(
            f"data/plotting/ccn-sig-file-{filename}-{key}.csv", index=False
        )

    return


def main():

    ##### Encoding Results Folder #####
    format = {
        "whisper-en-24": "/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-encoder-new-lag5k-25-all-24/*/*_comp.csv",
        "whisper-de-14": "/scratch/gpfs/kw1166/247-encoding/results/podcast/kw-podcast-full-777-whisper-medium.en-decoder-lag5k-25-all-14/*/*_comp.csv",
    }

    df = pd.read_csv("data/plotting/elec_masterlist.csv")
    df.dropna(subset="subject", inplace=True)
    df["subject"] = df.subject.astype(int).astype(str)
    df["elec"] = df.subject + "_" + df.name

    sig_elecs = {
        "allnodepth": df.loc[df.type != "D", "elec"].tolist(),
    }

    cors = []
    for key2, _ in format.items():
        layer = int(key2[-2:])
        type = key2[8:10]
        cors = read_folder(
            cors, format[key2], sig_elecs, "allnodepth", label=layer, type=type
        )
    cors = pd.concat(cors)
    cors.loc[:, "sig_mean"] = 0
    cors.loc[:, "sig_max"] = 0

    cors_en = cors[cors.type == "en"]
    cors_de = cors[cors.type == "en"]

    # get encoder sig elecs
    en_thresh = [0.060870, 0.042052, 0.071486, 0.112763]
    print(f"en_mean: {np.mean(en_thresh)}, en_max: {np.max(en_thresh)}")
    cors_en.loc[
        cors_en.loc[:, 0:400].max(axis=1) >= np.mean(en_thresh), "sig_mean"
    ] = 1
    cors_en.loc[
        cors_en.loc[:, 0:400].max(axis=1) >= np.max(en_thresh), "sig_max"
    ] = 1

    # get decoder sig elecs
    de_thresh = [0.075107, 0.046939, 0.078501, 0.091205]
    de_thresh2 = [0.091954, 0.046288, 0.086732, 0.098400]
    de_thresh_mean = max(np.mean(de_thresh), np.mean(de_thresh2))
    de_thresh_max = max(np.max(de_thresh), np.max(de_thresh2))
    print(f"de_mean: {de_thresh_mean}, de_max: {de_thresh_max}")
    cors_de.loc[
        cors_de.loc[:, 0:400].max(axis=1) >= de_thresh_mean, "sig_mean"
    ] = 1
    cors_de.loc[
        cors_de.loc[:, 0:400].max(axis=1) >= de_thresh_max, "sig_max"
    ] = 1

    # merge back to df
    df_new = df.loc[
        :, ("subject", "name", "elec", "princeton_class", "NYU_class2", "part160")
    ]
    cors_en = cors_en.loc[:, ("electrode", "sig_mean", "sig_max")]
    cors_de = cors_de.loc[:, ("electrode", "sig_mean", "sig_max")]
    df_new = df_new.merge(cors_en, left_on="elec", right_on="electrode")
    df_new.rename(
        columns={"sig_mean": "whisper_en_mean", "sig_max": "whisper_en_max"},
        inplace=True,
    )
    df_new.drop(columns="electrode", inplace=True)

    df_new = df_new.merge(cors_de, left_on="elec", right_on="electrode")
    df_new.rename(
        columns={"sig_mean": "whisper_de_mean", "sig_max": "whisper_de_max"},
        inplace=True,
    )
    df_new.drop(columns="electrode", inplace=True)

    # gpt2 significant
    gpt2_sig = pd.read_csv(
        "/scratch/gpfs/kw1166/247-plotting/data/plotting/podcast-old/164-phase-5000-sig-elec-gpt2xl50d-perElec-FDR-01-LH.csv"
    )
    gpt2_sig.loc[:, "elec"] = (
        gpt2_sig.subject.astype(str) + "_" + gpt2_sig.electrode
    )
    gpt2_sig.loc[:, "gpt2"] = 1
    df_new = df_new.merge(gpt2_sig.loc[:, ("elec", "gpt2")], how="left")
    df_new[["gpt2"]] = df_new[["gpt2"]].fillna(value=0).astype(int)

    # save sig elecs
    df_new.rename(columns={"name": "electrode"}, inplace=True)
    en_mean = df_new.loc[df_new.whisper_en_mean == 1, :]
    en_max = df_new.loc[df_new.whisper_en_max == 1, :]
    de_mean = df_new.loc[df_new.whisper_de_mean == 1, :]
    de_max = df_new.loc[df_new.whisper_de_max == 1, :]
    gpt2 = df_new.loc[df_new.gpt2 == 1, :]

    save_file(en_mean, "whisper-en-mean")
    save_file(en_max, "whisper-en-max")
    save_file(de_mean, "whisper-de-mean")
    save_file(de_max, "whisper-de-max")
    save_file(gpt2, "gpt2-medium")

    return


if __name__ == "__main__":
    main()
