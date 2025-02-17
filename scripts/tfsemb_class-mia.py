import os
import argparse
import sys

import csv
import pickle
import numpy as np
import pandas as pd
import string
import nltk

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tfsplt_utils import load_pickle
import statsmodels.api as sm

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


def ave_emb_wordlevel(datum):
    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["word"], sort=False)["embeddings"].apply(
        lambda x: mean_emb(x)
    )
    mean_embs = pd.DataFrame(mean_embs)
    mean_embs.reset_index(drop=True, inplace=True)

    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.drop_duplicates(["word"], inplace=True, ignore_index=True)
    datum2.loc[:, "embeddings"] = mean_embs["embeddings"]
    datum = datum2  # reassign back to datum

    return datum


def aggregate_df_mia():
    label_pkl = "data/pickling/podcast/777/pickles/777_full_labels.pkl"

    with open(label_pkl, "rb") as fh:
        datum = pickle.load(fh)
    datum = pd.DataFrame.from_dict(datum["labels"])
    return datum


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


def process_df(df, args):
    print(f"Original Datum Len: {len(df)}")
    df = add_speech(df)
    df = add_phoneme(df, args.loaddir)
    print(f"Total Phoneme #: {len(df)}")
    df = add_phoneme_cat(df, args.loaddir)
    df = add_phoneme_emb(df)

    return df


def logistic(df, x, y):
    print(f"Logistic from {x} to {y}")

    print(f"original # {len(df[y].unique())}")
    g = df.groupby(df[y])
    df = g.filter(lambda x: len(x) >= 10)
    df.reset_index(drop=True, inplace=True)
    print(f"new # {len(df[y].unique())}")

    kfolds = 10
    skf = KFold(n_splits=kfolds, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(len(df)))]

    if x in df.columns:  # logistic
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            LogisticRegression(max_iter=1000),
        )
    elif x == "freq":  # control 1
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            DummyClassifier(strategy="most_frequent"),
        )
        x = "embeddings"
    elif x == "uniform":  # control 2
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            DummyClassifier(strategy="uniform"),
        )
        x = "embeddings"
    elif x == "strat":  # control 3
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            DummyClassifier(strategy="stratified"),
        )
        x = "embeddings"

    scores = []
    scores.append(len(df[y].unique()))
    for i in range(kfolds):
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]
        test_index = folds[test_fold]
        train_index = np.concatenate([folds[j] for j in train_folds])

        X_train = df.loc[train_index, x]
        X_test = df.loc[test_index, x]
        Y_train = df.loc[train_index, y]
        Y_test = df.loc[test_index, y]

        X_train = np.array(X_train.tolist())
        X_test = np.array(X_test.tolist())
        Y_train = np.array(Y_train.tolist())
        Y_test = np.array(Y_test.tolist())

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        df.loc[test_index, "pred"] = preds

        # scores.append(model.score(X_test, Y_test))
        scores.append(balanced_accuracy_score(Y_test, preds))

    # score = sum(df.pred == df[y]) / len(df)
    score = balanced_accuracy_score(df[y], df.pred)
    scores.append(score)
    print(f"Prediction Accuracy: {score}")

    return scores


def classify(df, args):
    plot_dict = {
        "pho": "phoneme",
        "place_artic": "place_of_articulation",
        "manner_artic": "manner_of_articulation",
        "part_of_speech": "part_of_speech",
        # "voice": "voice_or_voiceless",
        # "function_content": "function_or_content",
    }
    if args.layer == 0:
        embs = [
            "embeddings",
            "freq",
            "uniform",
            "strat",
        ]
    else:
        embs = ["embeddings"]

    for dim in args.pca_dims:  # loop over dim
        ###### Classification ######
        results_df = pd.DataFrame(columns=[np.arange(0, 12, 1)])
        print(dim)
        for plot in plot_dict.keys():
            for emb in embs:
                scores = logistic(df, emb, plot)
                name = f"{emb}-{plot}"
                results_df.loc[name, :] = scores
        results_df.to_csv(os.path.join(args.savedir, args.save_csv))
        ###### Sig Test ######
        # rep = 1000
        # scores = logistic_sig(df, args.xcol, args.ycol, rep)
        # filename = os.path.join(args.savedir, f"b{args.xcol}_{args.ycol}_{rep}.csv")
        # with open(filename, "w") as csvfile:
        #     print("writing file")
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(scores)


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    df["is_nan"] = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~df["is_nan"]]

    return df


def ave_emb(datum):
    print("Averaging embeddings across tokens")

    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "embeddings"
    ].apply(lambda x: mean_emb(x))
    mean_embs = pd.DataFrame(mean_embs)

    # replace embeddings
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]
    mean_embs.set_index(datum.index, inplace=True)
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings"] = mean_embs.embeddings
    datum = datum2  # reassign back to datum

    return datum


def load_embs(args):
    base_datum = load_pickle(args.base_df)
    base_df = pd.DataFrame.from_dict(base_datum)
    emb_datum = load_pickle(args.emb_df)
    emb_df = pd.DataFrame.from_dict(emb_datum)

    base_df.reset_index(drop=False, inplace=True)
    df = pd.concat([base_df, emb_df], axis=1)

    df = df[df.adjusted_onset.notna()]
    df = drop_nan_embeddings(df)
    df = remove_punctuation(df)
    df.loc[df.is_nonword == False, :]
    df = ave_emb(df)
    return df


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true", default=False)
    parser.add_argument(
        "--savedir", nargs="?", type=str, default="results/paper-whisper"
    )
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    args.loaddir = "data/pickling/podcast/"
    aggr_file = f"podcast-categories.pkl"
    args.aggr_file = os.path.join(args.savedir, aggr_file)

    args.base_df = (
        f"data/pickling/podcast/777/pickles/embeddings/{args.emb}/full/base_df.pkl"
    )
    args.emb_df = f"data/pickling/podcast/777/pickles/embeddings/{args.emb}/full/cnxt_{args.context:04d}/layer_{args.layer:02d}.pkl"

    args.pca_dims = [50]
    args.save_csv = (
        f"classifier_pca50_{args.emb}_C{args.context:04d}_L{args.layer:02d}.csv"
    )

    # make save folder if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    return args


def main():
    args = arg_parser()

    # print(
    #     f"Aggregate: {args.aggregate}\t\tUsing en-{args.en_layer} and de-{args.de_layer}, {args.aggr_type} embeddings\n"
    # )
    # print(f"Classifier: {args.classify}\n")/

    # Aggregate or load file
    if args.aggregate:
        df = aggregate_df_mia()
        df = process_df(df, args)
        save_pickle(df, args.aggr_file)
    else:
        df = load_pickle(args.aggr_file)

    df2 = load_embs(args)
    df = pd.merge(
        df,
        df2.loc[:, ("word", "onset", "embeddings")],
        on=["word", "onset"],
        how="inner",
    )
    df = ave_emb_wordlevel(df)

    # Classifer
    classify(df, args)
    
    return


if __name__ == "__main__":
    main()
