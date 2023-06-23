import pickle
import os
import argparse
import sys

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tfsplt_utils import load_pickle

import nltk

# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("universal_tagset")


def run_pca(pca_to, df, col):
    print(f"PCA {col} to {pca_to}")
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df[col]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df[col] = pca_output.tolist()

    return df


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


def ave_emb(datum):
    print("Averaging embeddings across tokens")

    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["adjusted_onset", "word"], sort=False)["en_emb"].apply(
        lambda x: mean_emb(x)
    )
    mean_embs = pd.DataFrame(mean_embs)
    mean_embs2 = datum.groupby(["adjusted_onset", "word"], sort=False)["de_emb"].apply(
        lambda x: mean_emb(x)
    )
    mean_embs2 = pd.DataFrame(mean_embs2)

    # replace embeddings
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]
    mean_embs.set_index(datum.index, inplace=True)
    mean_embs2.set_index(datum.index, inplace=True)

    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "en_emb"] = mean_embs["en_emb"]
    datum2.loc[:, "de_emb"] = mean_embs2["de_emb"]
    datum = datum2  # reassign back to datum

    return datum


def ave_emb_wordlevel(datum):
    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["word"], sort=False)["en_emb"].apply(
        lambda x: mean_emb(x)
    )
    mean_embs = pd.DataFrame(mean_embs)
    mean_embs.reset_index(drop=True, inplace=True)
    mean_embs2 = datum.groupby(["word"], sort=False)["de_emb"].apply(
        lambda x: mean_emb(x)
    )
    mean_embs2 = pd.DataFrame(mean_embs2)
    mean_embs2.reset_index(drop=True, inplace=True)

    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.drop_duplicates(["word"], inplace=True, ignore_index=True)
    datum2.loc[:, "en_emb"] = mean_embs["en_emb"]
    datum2.loc[:, "de_emb"] = mean_embs2["de_emb"]
    datum = datum2  # reassign back to datum

    return datum


def aggregate_df(layer):
    print("Aggregating Data")

    # Arguments
    subjects = [625, 676, 7170, 798]
    dirname = "data/pickling/tfs/"
    # subjects = [777]
    # dirname = "data/pickling/podcast/"

    if layer == "":
        layer_en = "04"
        layer_de = "03"
    else:
        layer_en = int(layer)
        layer_de = int(layer)

    whisper_en = f"pickles/embeddings/whisper-tiny.en-encoder/full/cnxt_0001/layer_{layer_en:02}.pkl"
    whisper_de = f"pickles/embeddings/whisper-tiny.en-decoder/full/cnxt_0001/layer_{layer_de:02}.pkl"
    base_df_filename = "pickles/embeddings/whisper-tiny.en-decoder/full/base_df.pkl"

    # whisper_en = f"pickles/embeddings/whisper-medium.en-encoder-new/full/cnxt_0001/layer_{layer_en:02}.pkl"
    # whisper_de = f"pickles/embeddings/whisper-medium.en-decoder/full/cnxt_0001/layer_{layer_de:02}.pkl"
    # base_df_filename = "pickles/embeddings/whisper-medium.en-decoder/full/base_df.pkl"

    # load in data
    whisper_df = pd.DataFrame()

    for subj in subjects:
        temp_base_df = load_pickle(f"{dirname}{subj}/{base_df_filename}")
        temp_base_df = temp_base_df.dropna(subset=["onset", "offset"])
        temp_base_df.reset_index(drop=True, inplace=True)

        temp_en_df = load_pickle(f"{dirname}{subj}/{whisper_en}")
        temp_de_df = load_pickle(f"{dirname}{subj}/{whisper_de}")

        temp_base_df = temp_base_df.assign(en_emb=temp_en_df.embeddings)
        temp_base_df = temp_base_df.assign(de_emb=temp_de_df.embeddings)

        whisper_df = pd.concat([whisper_df, temp_base_df])

    whisper_df = whisper_df.loc[
        :,
        (
            "adjusted_onset",
            "token_idx",
            "word",
            "word_freq_overall",
            "in_glove50",
            "en_emb",
            "de_emb",
        ),
    ]
    whisper_df = whisper_df[whisper_df["en_emb"].notna()]
    whisper_df = whisper_df[whisper_df["de_emb"].notna()]
    whisper_df = ave_emb(whisper_df)

    return whisper_df


def add_speech(whisper_df):
    # Get Part of Speech
    words_orig, part_of_speech = zip(*nltk.pos_tag(whisper_df.word, tagset="universal"))
    whisper_df = whisper_df.assign(part_of_speech=part_of_speech)

    # Get function content
    function_content_dict = {
        "ADP": "function",
        "CONJ": "function",
        "DET": "function",
        "PRON": "function",
        "PRT": "function",
        "ADJ": "content",
        "ADV": "content",
        "NOUN": "content",
        "NUM": "content",
        "VERB": "content",
        "X": "unknown",
    }
    function_content = whisper_df.apply(
        lambda x: function_content_dict.get(x["part_of_speech"]), axis=1
    )
    whisper_df = whisper_df.assign(function_content=function_content)

    return whisper_df


def add_phoneme(whisper_df, dirname):
    # get phoneme dict
    cmu_dict_filename = f"{dirname}cmudict-0.7b"
    pdict = {}
    with open(cmu_dict_filename, "r", encoding="ISO-8859-1") as f:
        for line in f.readlines():
            if not line.startswith(";;;"):
                parts = line.rstrip().split()
                word = parts[0].lower()
                phones = [phone.rstrip("012") for phone in parts[1:]]
                pdict[word] = phones

    words2phonemes = whisper_df.apply(lambda x: pdict.get(x["word"].lower()), axis=1)

    # add to df
    whisper_df = whisper_df.assign(pho=words2phonemes)
    whisper_df = whisper_df[~whisper_df.pho.isnull()]
    whisper_df = whisper_df.explode("pho", ignore_index=False)
    whisper_df["pho_idx"] = (
        whisper_df.groupby(["word", "adjusted_onset"]).cumcount() + 1
    )

    return whisper_df


def add_phoneme_cat(whisper_df, dirname):
    # original categorization, including specific vowel catergorization
    # phoneset = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F' , 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L',  'M', 'N' , 'NG', 'OW', 'OY', 'P',  'R', 'S',  'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    # place_of_articulation   = ['low-central', 'low-front', 'mid-central', 'mid-back', 'high-back', 'high-front', 'bilabial', 'post-alveolar', 'alveolar', 'inter-dental', 'mid-front', 'mid-central', 'mid-front','alveolar','velar', 'glotal', 'high-front', 'high-front', 'post-alveolar', 'velar', 'alveolar', 'bilabial', 'alveolar', 'velar', 'high-back', 'high-front', 'bilabial', 'alveolar', 'alveolar', 'post-alveolar', 'alveolar', 'inter-dental', 'high-back', 'high-back', 'labio-dental', 'bilabial', 'palatal', 'alveolar', 'post-alveolar']
    # manner_of_articulation  = ['lax', 'lax', 'lax', 'lax', 'lax', 'tense', 'stop', 'affricate', 'stop', 'fricative', 'lax', 'tense', 'tense', 'flap', 'stop','fricative', 'lax', 'tense', 'affricate', 'stop', 'lateral-liquid', 'nasal', 'nasal', 'nasal', 'lax', 'lax', 'stop', 'retroflex-liquid', 'fricative', 'fricative', 'stop', 'fricative', 'lax', 'tense', 'fricative', 'glide', 'glide', 'fricative', 'fricative']

    # create categorizations
    phoneset_categorizations = pd.read_csv(f"{dirname}phoneset.csv")
    phoneset = phoneset_categorizations.Phoneme.values
    place_of_articulation = phoneset_categorizations.iloc[:, 1].values
    manner_of_articulation = phoneset_categorizations.iloc[:, 2].values
    voiced_or_voiceless = phoneset_categorizations.iloc[:, 3].values

    place_of_articulation_dict = dict(zip(phoneset, place_of_articulation))
    manner_of_articulation_dict = dict(zip(phoneset, manner_of_articulation))
    voiced_or_voiceless_dict = dict(zip(phoneset, voiced_or_voiceless))

    phocat = whisper_df.apply(
        lambda x: place_of_articulation_dict.get(x["pho"]), axis=1
    )
    whisper_df = whisper_df.assign(place_artic=phocat)
    phocat = whisper_df.apply(
        lambda x: manner_of_articulation_dict.get(x["pho"]), axis=1
    )
    whisper_df = whisper_df.assign(manner_artic=phocat)
    phocat = whisper_df.apply(lambda x: voiced_or_voiceless_dict.get(x["pho"]), axis=1)
    whisper_df = whisper_df.assign(voice=phocat)

    return whisper_df


def add_phoneme_emb(whisper_df):
    # select first few phonemes
    # whisper_df = whisper_df[whisper_df.pho_idx <= 4]
    print(f"First phonemes #: {sum(whisper_df.pho_idx == 1)}")
    print(f"Second phonemes #: {sum(whisper_df.pho_idx == 2)}")
    print(f"Third phonemes #: {sum(whisper_df.pho_idx == 3)}")
    print(f"Fourth phonemes #: {sum(whisper_df.pho_idx == 4)}")
    whisper_df = whisper_df[whisper_df.pho_idx == 1]

    # Get phoneme embeddings (for first phoneme)
    # const = 384
    # emb1 = []
    # for emb in whisper_df["en_emb"]:  # FIXME inefficient
    #     emb1.append(emb[0 : 3 * const])
    # whisper_df = whisper_df.assign(pho_emb=emb1)

    return whisper_df


def process_df(whisper_df, dirname, emb_type="ave"):
    print(f"Original Datum Len: {len(whisper_df)}")
    whisper_df = add_speech(whisper_df)

    # Get only meaningful content words
    # whisper_df = whisper_df[whisper_df.word_freq_overall < 10]

    if emb_type == "1st":
        print("Taking first instance of words")
        whisper_df.drop_duplicates(["word"], inplace=True, ignore_index=True)
    elif emb_type == "ave":
        print("Averaging embeddings across words")
        whisper_df = ave_emb_wordlevel(whisper_df)
    else:
        print("Keeping all words")
        pass

    print(f"After filtering: {len(whisper_df)}")

    whisper_df = add_phoneme(whisper_df, dirname)
    print(f"Total Phoneme #: {len(whisper_df)}")
    whisper_df = add_phoneme_cat(whisper_df, dirname)
    whisper_df = add_phoneme_emb(whisper_df)

    return whisper_df


def tsne(whisper_df, col):
    print(f"Doing t-SNE on {col}")
    tsne = TSNE(n_components=2, perplexity=50, random_state=329)
    embs = pd.DataFrame(np.vstack(whisper_df[col]))
    projections = pd.DataFrame(tsne.fit_transform(embs))
    return projections


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", nargs="?", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    # args = arg_parser()
    # layer = args.layer

    layer = ""
    emb_type = "1st"
    emb_type = "all"
    emb_type = "ave"

    loaddir = "data/pickling/tfs/"
    savedir = "results/20230612-whisper-tsne-no-filter"
    aggr_file = f"all4-whisper-embs-{emb_type}{layer}.pkl"
    aggr_file = os.path.join(savedir, aggr_file)
    tsne_file = f"all4-whisper-tsne-pca-{emb_type}{layer}.pkl"
    tsne_file = os.path.join(savedir, tsne_file)

    # Check if aggregate data exists. If not, aggregate
    if os.path.exists(aggr_file):
        df = load_pickle(aggr_file)
    else:
        df = aggregate_df(layer)
        df = process_df(df, loaddir, emb_type)
        save_pickle(df, aggr_file)

    # Check if tsne file exists. If not, make it
    if os.path.exists(tsne_file):
        pass
    else:
        # df = process_df(df, loaddir)
        df = run_pca(50, df, "en_emb")
        df = run_pca(50, df, "de_emb")
        tsne_en = tsne(df, "en_emb")
        tsne_de = tsne(df, "de_emb")
        df = df.reset_index(drop=True)
        df = df.assign(
            en_x=tsne_en[0],
            en_y=tsne_en[1],
            de_x=tsne_de[0],
            de_y=tsne_de[1],
        )
        save_pickle(df, tsne_file)


if __name__ == "__main__":
    main()
