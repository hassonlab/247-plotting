import os
import glob
import string

import numpy as np
import pandas as pd
import string
from scipy.spatial import distance
from scipy import stats


def main():

    convos = sorted(
        glob.glob(
            os.path.join(
                "results/tfs/kw-tfs-full-7170-glove50-lag10k-25-alll", "*", "*"
            )
        )
    )
    convos2 = sorted(
        glob.glob(
            os.path.join(
                "results/tfs/kw-tfs-full-7170-glove50-lag60k-10k-all", "*", "*"
            )
        )
    )

    convos = [os.path.basename(convo) for convo in convos]
    convos2 = [os.path.basename(convo) for convo in convos2]

    breakpoint()

    return


if __name__ == "__main__":
    main()
