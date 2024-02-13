import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("significant_electrodes_accoustic.csv")

    for mode in df.model.unique():
        if "non_aligned" not in mode:
            continue
        if "speech_embedding" in mode:
            tag = "last"
        else:
            tag = "first"

        df_part = df[df.model == mode]
        for patient in [625, 676, 7170, 798]:
            df_part_part = df_part[df.patient == patient]

            df_comp = df_part_part[df.comp_significant]
            new_df = pd.DataFrame({"subject": patient, "electrode": df_comp.electrode})
            filename = f"tfs-sig-file-{patient}-whisper-ac-{tag}-0.01-comp.csv"
            new_df.to_csv(filename, index=False)

            df_prod = df_part_part[df.prod_significant]
            new_df = pd.DataFrame({"subject": patient, "electrode": df_prod.electrode})
            filename = f"tfs-sig-file-{patient}-whisper-ac-{tag}-0.01-prod.csv"
            new_df.to_csv(filename, index=False)
    breakpoint()

    sid = "798"
    mode = "comp"
    file_name = f"tfs-sig-file-{sid}-whisper-en-first-0.01-{mode}.csv"
    df = pd.DataFrame({"subject": sid, "electrode": elec_list})
    # df.to_csv(file_name, index=False)

    return


if __name__ == "__main__":
    main()
