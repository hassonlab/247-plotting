import pandas as pd
import os

path = "/scratch/gpfs/ln1144/247-plotting"
os.chdir(path)

KEYS = ["prod", "comp"]

for key in KEYS:
    f1 = f'/scratch/gpfs/ln1144/247-plotting/results/cor-tfs-max/tfs_ave_whisper-en-last_{key}_sig.txt'
    f2 = f'/scratch/gpfs/ln1144/247-plotting/results/cor-tfs-max/tfs_ave_whisper-de-best_{key}_sig.txt'

    df_en = pd.read_csv(f1, names=["electrode","1","2","3","4","effect"])
    df_de = pd.read_csv(f2, names=["electrode","1","2","3","4","effect"])

    df_union = df_en.merge(df_de, how='outer')
    df_union.drop_duplicates(subset=["electrode","1","2","3","4"], keep='first', inplace=True)

    breakpoint()

    


