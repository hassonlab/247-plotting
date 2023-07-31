import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_base_df(sid, cor, emb_key):
    # Get brain coordinate file
    # coordinatefilename = f"data/plotting/brainplot/{sid}_{cor}_sig.txt"
    coordinatefilename = f"data/plotting/brainplot/{sid}_{cor}.txt"

    # data = pd.read_fwf(coordinatefilename, sep=" ", header=None)
    data = pd.read_csv(coordinatefilename, sep=" ", header=None)
    data = data.set_index(0)
    data = data.loc[:, 1:4]
    print(f"\nFor subject {sid}:\ntxt has {len(data.index)} electrodes")

    # Create filler columns
    for col in emb_key:
        data[col] = -1

    return data


def read_file(filename, path):
    # Read in one electrode encoding correlation results
    filename = os.path.join("data/encoding/", path, filename)
    if len(glob.glob(filename)) == 1:
        filename = glob.glob(filename)[0]
    elif len(glob.glob(filename)) == 0:
        return -1
    else:
        AssertionError("huh this shouldn't happen")
    elec_data = pd.read_csv(filename, header=None)
    return elec_data


def get_max(filename, path):
    # get max correlation for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    return max(elec_data.loc[0])


def get_max_before(filename, path):
    # get max correlation for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    half_idx = int(elec_data.shape[1] / 2)
    return max(elec_data.loc[0, :half_idx])


def get_max_after(filename, path):
    # get max correlation for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    half_idx = int(elec_data.shape[1] / 2)
    return max(elec_data.loc[0, half_idx:])


def get_maxidx(filename, path, lags):
    # get max correlation lag idx for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    return lags[elec_data.loc[0].idxmax()]


def get_lag(filename, path, lags, lag):
    # get correlation for specific lag for one electrode file
    elec_data = read_file(filename, path)
    if isinstance(elec_data, int):
        return -1
    (idx,) = np.where(lags == lag)
    return elec_data.loc[0, idx[0]]


def add_encoding(df, sid, formats, type="max", lags=[], chosen_lags=[]):
    for format in formats:
        # print(f"getting results for {format} embedding")
        for row, _ in df.iterrows():
            col_name2 = format + "_comp"
            comp_name = f"{row}_comp.csv"
            if type == "max":
                df.loc[row, col_name2] = get_max(comp_name, formats[format])
            elif type == "max_before":
                df.loc[row, col_name2] = get_max_before(comp_name, formats[format])
            elif type == "max_after":
                df.loc[row, col_name2] = get_max_after(comp_name, formats[format])
            elif type == "max_idx":
                df.loc[row, col_name2] = get_maxidx(comp_name, formats[format], lags)
            elif type == "lags":
                for lag in chosen_lags:
                    df.loc[row, f"{col_name2}_{lag}"] = get_lag(
                        comp_name, formats[format], lags, lag
                    )
            # elif type == "area":
            #     df.loc[row, col_name2] = get_area(comp_name, formats[format], lags, chosen_lags)

    return df


def save_file(df, sid, emb_keys, dir, cor, project):
    df.loc[:, 0] = df.index

    # max correlation
    for col in emb_keys:
        sid_file = os.path.join(dir, f"{sid}_{cor}_{col}.txt")
        df_output = df.loc[:, [0, 1, 2, 3, 4, col]]
        df_output.dropna(inplace=True)
        with open(sid_file, "w") as outfile:
            df_output.to_string(outfile, index=False, header=False)

    # slope
    # df_reg = df.iloc[:, 5:-1]
    # X = np.arange(1, 25).reshape((-1, 1))

    # def get_slope(y):
    #     model = LinearRegression().fit(X, y)
    #     return model.coef_[0]

    # df["slope"] = df_reg.apply(get_slope, axis=1) * 100

    # sid_file = os.path.join(dir, f"{sid}_{cor}_slope.txt")
    # with open(sid_file, "w") as outfile:
    #     df_output = df.loc[:, [0, 1, 2, 3, 4, "slope"]]
    #     df_output.to_string(outfile, index=False, header=False)

    # ratio / selectivity
    # df["04-24"] = (
    #     1 - df["whisper-encoder-04_comp"] / df["whisper-encoder-24_comp"]
    # )
    # df["01-24"] = (
    #     1 - df["whisper-encoder-01_comp"] / df["whisper-encoder-24_comp"]
    # )
    # df["24-04"] = df["whisper-encoder-24_comp"] - df["whisper-encoder-04_comp"]
    # df["24-01"] = df["whisper-encoder-24_comp"] - df["whisper-encoder-01_comp"]

    # selectivity = ["04-24", "01-24", "24-04", "24-01"]
    # for sel in selectivity:
    #     sid_file = os.path.join(dir, f"{sid}_{cor}_{sel}.txt")
    #     df_output = df.loc[:, [0, 1, 2, 3, 4, sel]]
    #     with open(sid_file, "w") as outfile:
    #         df_output.to_string(outfile, index=False, header=False)

    return


def main():
    ###### Core Arguments ######
    PRJ_ID = "podcast"
    sid = 777
    KEYS = ["comp"]
    COR_TYPE = "ave"  # average brain coordinates (for several patients)
    lags = np.arange(-5000, 5025, 25)
    select_lags = np.arange(-2000, 2025, 25)
    select_lags = [
        -600,
        # -550,
        # -500,
        # -450,
        -400,
        # -350,
        # -300,
        # -250,
        # -200,
        # -150,
        # -100,
        # -50,
        # 0,
        # 50,
        # 100,
        # 150,
        # 200,
        # 225,
        # 250,
        # 275,
        # 300,
        # 325,
        # 350,
        # 375,
        400,
        # 450,
        # 500,
        # 550,
        600,
    ]

    ##### Encoding Results Folder #####
    layers = [24]
    layers = np.arange(0, 25)
    layers = [0, 1, 8, 12, 16, 24]

    format = {}
    for layer in layers:
        format[
            f"whisper-encoder-{layer:02d}"
        ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230411-ccn/kw-podcast-full-777-whisper-medium.en-encoder-new-lag5k-25-all-{layer}/*/"
        # format[
        #     f"whisper-encoder-25"
        # ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230411-ccn/kw-podcast-full-777-whisper-medium.en-encoder-concat-lag5k-25-all/*/"
        # format[
        #     f"whisper-decoder-{layer:02d}"
        # ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230411-ccn/kw-podcast-full-777-whisper-medium.en-decoder-lag5k-25-all-{layer}/*/"
        # format[
        #     f"gpt-medium-n-1-{layer:02d}"
        # ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230409-gpt-medium-layers/kw-podcast-full-777-gpt2-medium-lag5k-25-all-{layer}/*/"
        # format[
        #     f"gpt-medium-{layer:02d}"
        # ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230409-gpt-medium-layers/kw-podcast-full-777-gpt2-medium-lag5k-25-all-shift-emb-{layer}/*/"
        # format[
        #     f"gpt-medium-118-{layer:02d}"
        # ] = f"/scratch/gpfs/kw1166/247-encoding/results/podcast/20230411-ccn/kw-podcast-full-777-gpt2-medium-lag5k-25-all-shift-emb-118-{layer}/*/"

    # Output directory name
    OUTPUT_DIR = "results/podcast-encoder-lags"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ##### Max correlation #####
    emb_key = [emb + "_" + key for emb in format.keys() for key in KEYS]
    df = get_base_df(sid, COR_TYPE, emb_key)  # get all electrodes

    # select only sig elecs
    sig_df = pd.read_csv("data/plotting/ccn-sig-file-whisper-en-max.csv")
    sig_df["elec"] = sig_df.subject.astype(str) + "_" + sig_df.electrode
    sig_df.set_index("elec", inplace=True)
    df = df.merge(sig_df, how="inner", left_index=True, right_index=True)
    df.drop(columns=["subject", "electrode"], inplace=True)
    # add encoding results
    emb_key = [f"{emb}_{lag}" for emb in emb_key for lag in select_lags]
    df = add_encoding(
        df, sid, format, "lags", lags, select_lags
    )  # add on the columns from encoding results
    save_file(df, sid, emb_key, OUTPUT_DIR, COR_TYPE, PRJ_ID)  # save txt files

    return


if __name__ == "__main__":
    main()
