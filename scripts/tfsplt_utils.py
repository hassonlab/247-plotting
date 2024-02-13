from distutils.command.config import dump_file
import glob
import os
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
from functools import partial
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def read_sig_file(filename, filedir, old_results=False):
    sig_file = pd.read_csv(os.path.join(filedir, filename))
    sig_file["sid_electrode"] = (
        sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
    )
    elecs = sig_file["sid_electrode"].tolist()

    if old_results:  # might need to use this for old 625-676 results
        elecs = sig_file["electrode"].tolist()  # no sid name in front

    return set(elecs)


def read_file(file_name, sigelecs, sigelecs_key, load_sid, key, label1, label2):
    elec = os.path.basename(file_name).replace(".csv", "")[:-5]
    if (  # Skip electrodes if they're not part of the sig list
        len(sigelecs)
        and elec not in sigelecs[sigelecs_key]
        # and "whole_brain" not in sigelecs_key
    ):
        return None
    # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
    #     continue
    df = pd.read_csv(file_name, header=None)
    # if df.max(axis=1)[0] < 0.1:
    #     return None
    df.insert(0, "sid", load_sid)
    df.insert(0, "key", key)
    df.insert(0, "electrode", elec)
    df.insert(0, "label1", label1)
    df.insert(0, "label2", label2)

    return df


def read_folder2(
    data,
    fname,
    load_sid="load_sid",
    label="label",
    mode="mode",
    type="all",
):
    files = glob.glob(fname)
    assert len(files) == 1, f"No files or multiple files found"
    df = pd.read_csv(files[0], header=None)
    df = df.dropna(axis=1)
    df.columns = np.arange(-1, 161)
    df = df.rename({-1: "electrode"}, axis=1)
    df.insert(1, "sid", load_sid)
    df.insert(1, "mode", mode)
    df.insert(0, "label", label)
    df.insert(0, "type", type)
    df.electrode = df.sid.astype(str) + "_" + df.electrode
    for i in np.arange(len(df)):
        data.append(df.iloc[[i]])
    return data


def read_folder(
    data,
    fname,
    sigelecs,
    sigelecs_key,
    load_sid="load_sid",
    key="key",
    label1="label1",
    label2="label2",
    parallel=True,
):
    files = glob.glob(fname)
    assert (
        len(files) > 0
    ), f"No results found under {fname}"  # check files exist under format

    if parallel:
        p = Pool(10)
        for result in p.map(
            partial(
                read_file,
                sigelecs=sigelecs,
                sigelecs_key=sigelecs_key,
                load_sid=load_sid,
                key=key,
                label1=label1,
                label2=label2,
            ),
            files,
        ):
            data.append(result)

    else:
        for resultfn in files:
            data.append(
                read_file(
                    resultfn,
                    sigelecs,
                    sigelecs_key,
                    load_sid,
                    key,
                    label1,
                    label2,
                )
            )

    return data


def load_pickle(file, key=None):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    if key:
        df = pd.DataFrame.from_dict(datum[key])
    else:
        df = pd.DataFrame.from_dict(datum)
    return df


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


def colorFader(c1, c2, mix):
    """Get color in between two colors (based on linear interpolate)

    Args:
        c1: color 1 in hex format
        c2: color 2 in hex format
        mix: percentage between two colors (0 is c1, 1 is c2)

    Returns:
        a color in hex format
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_cat_color(num=10):
    """Get categorical colors"""
    if num > 10:
        print("Can't get more than 10 categorical colors")
        return None
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]  # separate colors
    return colors


def get_con_color(colormap, num):
    """Get continuous colors"""
    cmap = plt.cm.get_cmap(colormap)
    colors = [cmap(i / num) for i in range(0, num)]
    return colors


def get_fader_color(c1, c2, num):
    """Get self-defined continuous colors"""
    colors = [colorFader(c1, c2, i / num) for i in range(0, num)]
    return colors


class Colormap2D(mpl.colors.Colormap):
    # https://gallantlab.org/pycortex/colormaps.html

    def __init__(
        self,
        cmap,
        vmin=None,
        vmax=None,
        vmin2=None,
        vmax2=None,
        hflip=False,
        vflip=False,
    ):
        I = plt.imread(f"data/plotting/cmaps/{cmap}.png")
        if hflip and vflip:
            I = np.flip(I, (0, 1))
        elif hflip:
            I = np.flip(I, 0)
        elif vflip:
            I = np.flip(I, 1)

        cmap_new = mpl.colors.ListedColormap(np.squeeze(I))
        plt.cm.register_cmap(cmap, cmap_new)
        self.cmap = cmap_new
        self.vmin = vmin
        self.vmax = vmax
        self.vmin2 = vmin if vmin2 is None else vmin2
        self.vmax2 = vmax if vmax2 is None else vmax2
        N = self.cmap.colors.shape[0]
        super().__init__(cmap, N)

    def __call__(self, X, alpha=None, bytes=False):
        data1 = X[:, 0]
        data2 = X[:, 1]

        cmap = self.cmap.colors

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)

        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1] - 1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32)
        dim2 = np.round(d2 * (cmap.shape[0] - 1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # map r, g, b, a values between 0 and 255 to avoid problems with
        # VolumeRGB when plotting flatmaps with quickflat
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = colored.T
        r.shape = dim1.shape
        g.shape = dim1.shape
        b.shape = dim1.shape
        a.shape = dim1.shape
        # Preserve nan values as alpha = 0
        aidx = np.logical_or(np.isnan(data1), np.isnan(data2))
        a[aidx] = 0
        return r, g, b, a

    def __hash__(self):
        return hash(self.name)
