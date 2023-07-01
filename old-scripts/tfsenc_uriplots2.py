import os
import glob

import numpy as np
import pandas as pd


def read_file(file_name):
    df = pd.read_csv(file_name, header=None)

    return df


def main():
    result_path = "/tigress/kw1166/247-encoding-results/13uriplots2/data-new"
    # parent_divs = glob.glob("results/uriplot2/data/*")
    parent_divs = glob.glob(
        "/tigress/kw1166/247-encoding-results/13uriplots2/data/*"
    )
    for parent_div in parent_divs:
        sid_idx = os.path.basename(parent_div).find("-")
        sid = os.path.basename(parent_div)[0:sid_idx]
        subname = "247-" + os.path.basename(parent_div)[sid_idx + 1 :]
        result_subpath = os.path.join(result_path, subname)
        if not (os.path.exists(result_subpath)):
            os.mkdir(result_subpath)  # make new directory if not exist

        df = read_file(parent_div + "/results-comp.csv")
        for i in np.arange(0, len(df)):
            df2 = df.iloc[[i], 1:162]
            elec_name = df.iloc[i, 0]
            filename = sid + "_" + elec_name + "_comp.csv"
            df2.to_csv(
                os.path.join(result_subpath, filename),
                index=False,
                header=False,
            )
        df = read_file(parent_div + "/results-prod.csv")
        for i in np.arange(0, len(df)):
            df2 = df.iloc[[i], 1:162]
            elec_name = df.iloc[i, 0]
            filename = sid + "_" + elec_name + "_prod.csv"
            df2.to_csv(
                os.path.join(result_subpath, filename),
                index=False,
                header=False,
            )

    return None


if __name__ == "__main__":
    main()
