import os
import argparse
import sys

import csv
import pickle
import numpy as np
import pandas as pd
import nltk

from sklearn.manifold import TSNE
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


def aggregate_df(args):
    print("Aggregating Data")

    subjects = [625, 676, 7170, 798]
    subjects = [777]

    # whisper_en = f"pickles/embeddings/whisper-tiny.en-encoder/full/cnxt_0001/layer_{args.en_layer:02}.pkl"
    # whisper_de = f"pickles/embeddings/whisper-tiny.en-decoder/full/cnxt_0001/layer_{args.de_layer:02}.pkl"
    # base_df_filename = "pickles/embeddings/whisper-tiny.en-decoder/full/base_df.pkl"
    whisper_en = f"pickles/embeddings/whisper-tiny.en-encoder-replicate/full/cnxt_0001/layer_{args.en_layer:02}.pkl"
    whisper_de = f"pickles/embeddings/whisper-tiny.en-decoder-replicate/full/cnxt_0001/layer_{args.de_layer:02}.pkl"
    base_df_filename = (
        "pickles/embeddings/whisper-tiny.en-decoder-replicate/full/base_df.pkl"
    )

    # load in data
    whisper_df = pd.DataFrame()

    for subj in subjects:
        temp_base_df = load_pickle(f"{args.loaddir}{subj}/{base_df_filename}")
        temp_base_df = temp_base_df.dropna(subset=["onset", "offset"])
        temp_base_df.reset_index(drop=True, inplace=True)

        temp_en_df = load_pickle(f"{args.loaddir}{subj}/{whisper_en}")
        temp_de_df = load_pickle(f"{args.loaddir}{subj}/{whisper_de}")

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


def aggregate_df_daria(args):
    print("Aggregating Data")

    whisper_en = "data/pickling/podcast/999/speech_embeddings.pickle"
    whisper_de = "data/pickling/podcast/999/lang_embeddings.pickle"
    whisper_base = "data/pickling/podcast/999/word_onset_offset_df.pickle"

    with open(whisper_base, "rb") as fh:
        datum = pickle.load(fh)
    with open(whisper_en, "rb") as fh:
        whisper_en = pickle.load(fh)
    with open(whisper_de, "rb") as fh:
        whisper_de = pickle.load(fh)
    datum = pd.DataFrame(datum)
    whisper_en = [en[4][0].tolist() for en in whisper_en]
    whisper_de = [de[3][0].tolist() for de in whisper_de]
    datum["en_emb"] = whisper_en
    datum["de_emb"] = whisper_de
    datum["adjusted_onset"] = datum.audio_onset
    datum["adjusted_offset"] = datum.audio_offset
    datum["onset"] = datum.audio_onset
    datum["offset"] = datum.audio_offset

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


def process_df(whisper_df, args):
    print(f"Original Datum Len: {len(whisper_df)}")
    whisper_df = add_speech(whisper_df)

    # Get only meaningful content words
    # whisper_df = whisper_df[whisper_df.word_freq_overall < 10]

    if args.aggr_type == "1st":
        print("Taking first instance of words")
        whisper_df.drop_duplicates(["word"], inplace=True, ignore_index=True)
    elif args.aggr_type == "ave":
        print("Averaging embeddings across words")
        whisper_df = ave_emb_wordlevel(whisper_df)
    else:
        print("Keeping all words")
        pass

    print(f"After filtering: {len(whisper_df)}")

    whisper_df = add_phoneme(whisper_df, args.loaddir)
    print(f"Total Phoneme #: {len(whisper_df)}")
    whisper_df = add_phoneme_cat(whisper_df, args.loaddir)
    whisper_df = add_phoneme_emb(whisper_df)

    return whisper_df


def tsne(whisper_df, col):
    print(f"Doing t-SNE on {col}")
    tsne = TSNE(n_components=2, perplexity=50, random_state=329)
    embs = pd.DataFrame(np.vstack(whisper_df[col]))
    projections = pd.DataFrame(tsne.fit_transform(embs))
    return projections


def logistic_sig(df, x, y, shuffle_num=5):
    print(f"Logistic from {x} to {y}")
    print(f"original # {len(df[y].unique())}")
    g = df.groupby(df[y])
    df = g.filter(lambda x: len(x) >= 100)
    df.reset_index(drop=True, inplace=True)
    print(f"new # {len(df[y].unique())}")

    kfolds = 10
    skf = KFold(n_splits=kfolds, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(len(df)))]

    model = make_pipeline(
        StandardScaler(),
        # PCA(50, whiten=True),
        LogisticRegression(max_iter=1000),
    )

    # PCA early for x, shuffle for y
    df = run_pca(50, df, x)

    scores = []
    for rep in np.arange(0, shuffle_num):
        print(f"Rep {rep}")
        df[y] = np.random.permutation(df[y].values)  # shuffle y
        for i in range(kfolds):
            # print(f"\t Fold {i}")
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
        score = balanced_accuracy_score(df[y], df.pred)
        # score = accuracy_score(df[y], df.pred)
        scores.append(score)

    # Sig test code that didn't work
    # pca = PCA(n_components=50, svd_solver="auto", whiten=True)
    # pca_output = pca.fit_transform(X_train)
    # estimator = LogisticRegression()
    # score, permutation_scores, pvalue = permutation_test_score(
    #     model, X_train, Y_train, cv=10, random_state=0
    # )
    # More code that didn't work
    # sm_model = sm.Logit(Y_train, sm.add_constant(X_train)).fit(disp=0)
    # logit_model = sm.Logit(Y_train.tolist(), X_train.tolist(), missing="drop")
    # result = logit_model.fit()
    # print(result.summary())

    return scores


def logistic(df, x, y):
    print(f"Logistic from {x} to {y}")

    print(f"original # {len(df[y].unique())}")
    g = df.groupby(df[y])
    df = g.filter(lambda x: len(x) >= 5)
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
        x = "de_emb"
    elif x == "uniform":  # control 2
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            DummyClassifier(strategy="uniform"),
        )
        x = "de_emb"
    elif x == "strat":  # control 3
        model = make_pipeline(
            StandardScaler(),
            PCA(50, whiten=True),
            DummyClassifier(strategy="stratified"),
        )
        x = "de_emb"

    scores = []
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
    embs = [
        "en_emb",
        "de_emb",
        "freq",
        "uniform",
        "strat",
    ]

    for dim in args.pca_dims:  # loop over dim
        ###### Classification ######
        results_df = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        print(dim)
        for plot in plot_dict.keys():
            for emb in embs:
                scores = logistic(df, emb, plot)
                name = f"{emb}-{plot}"
                results_df.loc[name, :] = scores
        results_df.to_csv(
            os.path.join(
                args.savedir,
                f"classifier_pca{dim}_filter-100_{args.aggr_type}_L{args.layer}.csv",
            )
        )
        ###### Sig Test ######
        # rep = 1000
        # scores = logistic_sig(df, args.xcol, args.ycol, rep)
        # filename = os.path.join(args.savedir, f"b{args.xcol}_{args.ycol}_{rep}.csv")
        # with open(filename, "w") as csvfile:
        #     print("writing file")
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(scores)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", nargs="?", type=str, default="")
    parser.add_argument("--aggregate", action="store_true", default=False)
    parser.add_argument("--tsne", action="store_true", default=False)
    parser.add_argument("--pca", action="store_true", default=False)
    parser.add_argument("--classify", action="store_true", default=False)
    parser.add_argument("--aggr-type", nargs="?", type=str, default="ave")
    parser.add_argument(
        "--savedir", nargs="?", type=str, default="results/paper-whisper"
    )
    parser.add_argument("--xcol", nargs="?", type=str, default="")
    parser.add_argument("--ycol", nargs="?", type=str, default="")
    args = parser.parse_args()

    args.en_layer = int(args.layer)
    args.de_layer = int(args.layer)

    if args.layer == "-1":
        args.layer = ""
        args.en_layer = 4
        args.de_layer = 3

    # args.loaddir = "data/pickling/tfs/"
    # aggr_file = f"all4-whisper-embs-{args.aggr_type}{args.layer}.pkl"
    # args.aggr_file = os.path.join(args.savedir, aggr_file)
    # tsne_file = f"all4-whisper-tsne-{args.aggr_type}{args.layer}.pkl"
    # args.tsne_file = os.path.join(args.savedir, tsne_file)
    # pca_file = f"all4-whisper-pca-{args.aggr_type}{args.layer}.pkl"
    # args.pca_file = os.path.join(args.savedir, pca_file)

    args.loaddir = "data/pickling/podcast/"
    aggr_file = f"pod-whisper-embs-{args.aggr_type}{args.layer}.pkl"
    args.aggr_file = os.path.join(args.savedir, aggr_file)
    tsne_file = f"pod-whisper-tsne-{args.aggr_type}{args.layer}.pkl"
    args.tsne_file = os.path.join(args.savedir, tsne_file)
    pca_file = f"pod-whisper-pca-{args.aggr_type}{args.layer}.pkl"
    args.pca_file = os.path.join(args.savedir, pca_file)

    args.pca_dims = [50]

    # make save folder if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    return args


def main():
    args = arg_parser()

    print(
        f"Aggregate: {args.aggregate}\t\tUsing en-{args.en_layer} and de-{args.de_layer}, {args.aggr_type} embeddings\n"
    )
    print(f"t-SNE: {args.tsne}\n")
    print(f"Classifier: {args.classify}\n")

    # Aggregate or load file
    if args.aggregate:
        df = aggregate_df_daria(args)
        df = process_df(df, args)
        save_pickle(df, args.aggr_file)
    else:
        df = load_pickle(args.aggr_file)

    # t-SNE
    if args.tsne:
        tsne_en = tsne(df, "en_emb")
        tsne_de = tsne(df, "de_emb")
        df = df.reset_index(drop=True)
        df = df.assign(
            en_x=tsne_en[0],
            en_y=tsne_en[1],
            de_x=tsne_de[0],
            de_y=tsne_de[1],
        )
        save_pickle(df, args.tsne_file)

    if args.pca:
        pca = PCA(n_components=2, svd_solver="auto", whiten=True)
        embs = np.vstack(df["en_emb"].values)
        pca_output = pca.fit_transform(embs)
        df["en_x"] = pca_output[:, 0]
        df["en_y"] = pca_output[:, 1]

        pca = PCA(n_components=2, svd_solver="auto", whiten=True)
        embs = np.vstack(df["de_emb"].values)
        pca_output = pca.fit_transform(embs)
        df["de_x"] = pca_output[:, 0]
        df["de_y"] = pca_output[:, 1]
        save_pickle(df, args.pca_file)
        breakpoint()
        # df[col] = pca_output.tolist()

    # Classifer
    if args.classify:
        classify(df, args)


if __name__ == "__main__":
    main()
