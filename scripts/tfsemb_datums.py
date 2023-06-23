import glob
import os
import pandas as pd
import numpy as np
import pickle

from tfsplt_utils import load_pickle


def main():

    sids = [625, 676, 7170, 798]

    all_df = pd.DataFrame()
    dfs = {}

    for sid in sids:
        datum_dir = f"data/pickling/tfs/{sid}/pickles/{sid}_full_labels.pkl"
        df = load_pickle(datum_dir, key="labels")

        df.drop(
            columns={
                "index",
                "fine_flag",
                "full_signal_length",
                "trimmed_signal_length",
            },
            errors="ignore",
            inplace=True,
        )

        dfs[sid] = df
        all_df = pd.concat([all_df, df], axis=0)

    dfs["all"] = all_df

    print("Number of words")
    for sid, df in dfs.items():
        print(f"Patient {sid}, Length of Datum {len(df)}, Prod {len(df.loc[df.production==1,:])}, Comp {len(df.loc[df.production==0,:])}")

    # Patient 625, Length of Datum 79652, Prod 32010, Comp 47642
    # Patient 676, Length of Datum 213944, Prod 103757, Comp 110187
    # Patient 7170, Length of Datum 117077, Prod 45653, Comp 71424
    # Patient 798, Length of Datum 99839, Prod 45356, Comp 54483
    # Patient all, Length of Datum 510512, Prod 226776, Comp 283736

    print("Average word length")
    for sid, df in dfs.items():
        word_len = (df.adjusted_offset - df.adjusted_onset) / 512 * 1000
        print(f"Patient {sid}, {word_len.describe()}")

    # Patient 625, count    79652.000000
    # mean       238.964434
    # std        150.926380
    # min       -595.782227
    # 25%        130.000000
    # 50%        201.171875
    # 75%        310.000000
    # max       3020.000100

    # Patient 676, count    213944.000000
    # mean        224.413673
    # std         147.663008
    # min       -5653.696094
    # 25%         130.000000
    # 50%         190.000000
    # 75%         280.000000
    # max        3230.000000

    # Patient 7170, count    117077.000000
    # mean        212.593868
    # std         161.784362
    # min       -3291.900000
    # 25%         110.000000
    # 50%         180.000000
    # 75%         270.000000
    # max        6640.000199

    # Patient 798, count    99839.000000
    # mean       242.387196
    # std        223.247987
    # min          0.000000
    # 25%        104.500000
    # 50%        180.500000
    # 75%        313.500000
    # max      18819.500000

    # Patient all, count    510512.000000
    # mean        227.488289
    # std         168.996793
    # min       -5653.696094
    # 25%         120.000000
    # 50%         189.453125
    # 75%         290.000000
    # max       18819.500000

    # 132 words with negative word length, 4 with zero word length

    all_all_nums = pd.DataFrame()
    for sid, df in dfs.items():
        if sid == "all":
            continue

        df["audio_onset"] = (df.onset + 3000) / 512
        df["audio_offset"] = (df.offset + 3000) / 512

        all_nums = pd.DataFrame()

        for conv_id in df.conversation_id.unique():
            # print(conv_id)
            df_conv = df.loc[df.conversation_id == conv_id, :]

            def count_context_words(audio_onset):
                chunk_offset = audio_onset + 0.2325
                chunk_onset = chunk_offset - 30
                return len(
                    df_conv[
                        (df_conv.audio_onset >= chunk_onset)
                        & (df_conv.audio_onset <= chunk_offset)
                    ]
                )

            nums = df_conv.audio_onset.apply(count_context_words)
            all_nums = pd.concat([all_nums, nums], axis=0)

        print(f"Patient {sid}, {all_nums.describe()}")
        all_all_nums = pd.concat([all_all_nums, all_nums], axis=0)
        all_all_nums.reset_index(drop=True, inplace=True)

    print(f"All Patients, {all_all_nums.describe()}")

    # Patient 625                                     
    # count  79652.000000
    # mean      61.748531
    # std       25.856134
    # min        1.000000
    # 25%       44.000000               
    # 50%       64.000000                                                                       
    # 75%       80.000000                                                                   
    # max      132.000000

    # Patient 676                                                   
    # count  213944.000000
    # mean       71.434399                                                                         
    # std        28.102933                
    # min         1.000000
    # 25%        52.000000
    # 50%        76.000000
    # 75%        92.000000
    # max       151.000000

    # Patient 7170
    # count  117077.000000
    # mean       77.876884
    # std        27.936912
    # min         1.000000
    # 25%        62.000000
    # 50%        81.000000
    # 75%        97.000000
    # max       157.000000

    # Patient 798
    # count  99839.000000
    # mean      64.281784
    # std       27.061226
    # min        1.000000
    # 25%       47.000000
    # 50%       69.000000
    # 75%       85.000000
    # max      170.000000

    # All Patients
    # count  510512.000000
    # mean       70.001833
    # std        28.099683
    # min         1.000000
    # 25%        51.000000
    # 50%        74.000000
    # 75%        91.000000
    # max       170.000000

    return


if __name__ == "__main__":
    main()
