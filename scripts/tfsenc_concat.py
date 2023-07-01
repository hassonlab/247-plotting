import argparse
import csv
import glob
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--lags", nargs="+", type=float, required=True)
parser.add_argument("--lags-final", nargs="+", type=float, required=True)
parser.add_argument("--output-dir", type=str, default="results/tfs/new-concat")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

files1 = glob.glob(args.formats[0] + "/*.csv")

formats = args.formats[1:]
for format in formats:
    files2 = glob.glob(format + "/*.csv")
    assert len(files1) == len(
        files2
    ), "Need same number files under data sources"

for file in files1:
    if "summary" in file:
        continue
    filename = os.path.basename(file)  # get filename
    newfilename = args.output_dir + filename

    df = pd.read_csv(file, header=None)
    for format in formats:
        file2 = format + filename
        df2 = pd.read_csv(file2, header=None)
        df = pd.concat([df, df2], axis=1)  # concatenate two dataframe

    df.columns = args.lags  # add lags
    df = df.reindex(columns=sorted(df.columns))  # sort columns by lags

    if len(args.lags_final) != 1:
        df = df[args.lags_final]  # select final lags to include

    with open(newfilename, "w") as csvfile:
        print(f"merging file {filename}")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([df.loc[0].tolist()])
