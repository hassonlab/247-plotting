import pandas as pd
import numpy as np


def main():
    file = "/home/kw1166/scratch/247/247-plotting/data/plotting/paper-prob-improb/sig_test_llama3_version1.csv"
    save_file = "/home/kw1166/scratch/247/247-plotting/data/plotting/paper-prob-improb/%s-llama3-sig-%s.csv"
    elecs = pd.read_csv(file, index_col=0)

    for sid in [625, 676, 7170, 798]:
        sid_elecs = elecs[elecs.patient == sid]
        prod = sid_elecs[sid_elecs.prod_significant]
        prod = pd.DataFrame({"subject": sid, "electrode": prod.electrode})
        prod.to_csv(save_file % (sid, "prod"), index=False)
        comp = sid_elecs[sid_elecs.comp_significant]
        comp = pd.DataFrame({"subject": sid, "electrode": comp.electrode})
        comp.to_csv(save_file % (sid, "comp"), index=False)

    return


if __name__ == "__main__":
    main()
