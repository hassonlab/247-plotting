import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tfsplt_utils import Colormap2D
import plotly.express as px


def main():
    filepath = "data/plotting/sig-elecs/20230413-whisper-paper"
    sids = [625, 676, 7170, 798]

    for sid in sids:
        for mode in ["prod", "comp"]:
            # en = pd.read_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-en-last-0.01-{mode}.csv"
            # )
            # de = pd.read_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-de-best-0.01-{mode}.csv"
            # )
            ac = pd.read_csv(
                f"{filepath}/tfs-sig-file-{sid}-whisper-ac-first-0.01-{mode}.csv"
            )
            ac2 = pd.read_csv(
                f"{filepath}/tfs-sig-file-{sid}-whisper-ac-last-0.01-{mode}.csv"
            )

            # ende = en.merge(de, how="outer")
            # enac = en.merge(ac, how="outer")
            # acde = ac.merge(de, how="outer")
            acac = ac.merge(ac2, how="outer")
            # ende.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-ende-outer-{mode}.csv",
            #     index=False,
            # )
            # enac.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-enac-outer-{mode}.csv",
            #     index=False,
            # )
            # acde.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-acde-outer-{mode}.csv",
            #     index=False,
            # )
            acac.to_csv(
                f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-acac-outer-{mode}.csv",
                index=False,
            )
            # ende = en.merge(de, how="inner")
            # enac = en.merge(ac, how="inner")
            # acde = ac.merge(de, how="inner")
            acac = ac.merge(ac2, how="inner")
            # ende.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-ende-inner-{mode}.csv",
            #     index=False,
            # )
            # enac.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-enac-inner-{mode}.csv",
            #     index=False,
            # )
            # acde.to_csv(
            #     f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-acde-inner-{mode}.csv",
            #     index=False,
            # )
            acac.to_csv(
                f"{filepath}/tfs-sig-file-{sid}-whisper-varpar-acac-inner-{mode}.csv",
                index=False,
            )

    # cc = Colormap2D(
    #     "PU_RdBu_covar", vmin=0, vmax=1, vmin2=0, vmax2=1
    # )  # GreenWhiteBlue_2D')
    # color_list = px.colors.qualitative.D3
    # cols = np.array([[0.5, 0.5], [0, 0], [0.9, 0.1], [0.1, 0.9]])
    # cols = np.array([[0.5, 0, 0.9, 0.1], [0.5, 0, 0.1, 0.9]])
    # red, green, blue, alpha = cc(cols)
    breakpoint()
    return


if __name__ == "__main__":
    main()
