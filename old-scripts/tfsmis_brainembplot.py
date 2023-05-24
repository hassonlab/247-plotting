# e22 - rerran 10 folds, pc
# e23 - same but 4s

import os
import pickle
import string
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from multiprocessing import Pool

from scipy.stats import zscore
from statsmodels.stats import multitest
from statsmodels.stats import stattools

from tfsenc_read_datum import load_datum
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


n_workers = 10
nperms = 5000
pdir = '/scratch/gpfs/zzada/247-encoding/results/podcast/'
pdir = '/scratch/gpfs/kw1166/zaid-247-encoding/results/podcast/'

dirs = ['0shot-zz-podcast-full-777-gpt2-xl-e23/777/',
        '0shot-zz-podcast-full-777-gpt2-xl-e23-sh/777/']

# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717l/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717lfdp/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717fdp/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717detrend/777/']
dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717ldp1/777/']

dirs = ['0shot-zz-podcast-full-777-gpt2-xl-polydet2-nnt/777/',
        '0shot-zz-podcast-full-777-gpt2-xl-polydet2-nnt-sh/777/']

# dirs = ['0shot-kw-podcast-full-777-glove50-polydet2-nnt/777/',
#         '0shot-kw-podcast-full-777-glove50-polydet2-nnt-sh/777/']

dirs = ['0shot-kw-podcast-full-777-gpt2-xl-polydet2-nnt/777/',
        '0shot-kw-podcast-full-777-gpt2-xl-polydet2-nnt-sh/777/']


# Create experiments from master list
elecs = pd.read_csv('/scratch/gpfs/zzada/247-encoding/data/elec_masterlist.csv')
cats = ['princeton_class', 'NYU_class']

cats = ['NYU_class']
subjects = [717, 798, 742, [717,798,742], [662,717,723,741,742,743,763,798]]
subjects = [717, 798, 742, [717,798,742]]
rois = ['IFG', 'precentral', 'postcentral', 'STG']
# rois = ['IFG', 'precentral']

experiments = {}
for category in cats:
    for subject in subjects:
        for roi in rois:
            if isinstance(subject, int):
                crit = elecs.subject == subject
                name = '_'.join([str(subject), category, roi])
            elif isinstance(subject, list):
                m = len(subject)
                crit = elecs.subject.isin(subject)
                name = '_'.join([f'all{m}', category, roi])
            crit &= (elecs[category] == roi)
            subdf = elecs[crit]
            es = [str(x) + '_' + y for x, y in zip(subdf.subject, subdf.name)]
            if len(es):
                experiments[name] = es

custom = dirs[0].split('/')[0][-3:]
print(custom)
print(len(experiments), 'experiments')

            # if elec == '717_LGB79':


def correlate(A, B, axis=0):
    """Calculate pearson correlation between two matricies.

       axis = 0 correlates columns in A to columns in B
       axis = 1 correlates rows in A to rows in B
    """
    assert A.ndim == B.ndim, 'Matrices must have same number of dimensions'
    assert A.shape == B.shape, 'Matrices must have same shape'

    A_mean = A.mean(axis=axis, keepdims=True)
    B_mean = B.mean(axis=axis, keepdims=True)
    A_stddev = np.sum((A - A_mean)**2, axis=axis)
    B_stddev = np.sum((B - B_mean)**2, axis=axis)

    num = np.sum((A - A_mean) * (B - B_mean), axis=axis)
    den = np.sqrt(A_stddev * B_stddev)

    return num / den


def one_samp_perm(x, nperms):
    n = len(x)
    dist = np.zeros(nperms)
    for i in range(nperms):
        dist[i] = np.random.choice(x, n, replace=True).mean()

    # s = np.sort(dist)  # unnecessary
    # val = np.sum(s > 0)
    val = np.sum(dist > 0)
    p_value = 1 - val / nperms
    # == np.mean(s < 0)
    return p_value


def paired_permutation(x, y, nperms):
    # Order of x and y matters
    n = len(x)
    truescore = (x - y).mean()
    dist = np.zeros(nperms)
    for i in range(nperms):
        s = np.random.choice([1, -1], n)
        dist[i] = np.mean(s * (x-y))

    p_value = (truescore > dist).mean()
    return p_value


def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(pvals,
                                        method='fdr_bh',
                                        is_sorted=False)
    return pcor


def run_exp(experiment, elecs, y_nn_idxs):
    print(experiment, len(elecs))

    dfs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(0, ls='-', c='black', alpha=0.1)
    ax.axhline(0, ls='-', c='black', alpha=0.1)
    ax.set(xlabel='Lag (s)', ylabel='Correlation (r+se)')
    # ax.set_ylim([-0.05, 0.25])

    for i, resultdir in enumerate(dirs):
        lags = []
        signal = []
        pred_signal = []
        nn_signal = []
        nnt_signal = []
        ynn_test_signal = []

        # Load results of this run
        for elec in elecs:
            if pd.isna(elec):
                continue
            # filename = pdir + resultdir + elec[2:5] + '_' + elec[30:] + '.pkl'
            filename = pdir + resultdir + elec + '.pkl'
            if not os.path.isfile(filename):
                # print(filename, 'is not found')
                continue
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                lags = data['lags']
                signal.append(data['Y_signal'])
                pred_signal.append(data['Yhat_signal'])
                nn_signal.append(data['Yhat_nn_signal'])
                nnt_signal.append(data['Yhat_nnt_signal'])
                ynn_test_signal.append(data['Y_signal'][y_nn_idxs])

        if len(signal) == 0:
            print('None of the electrodes were found')
            break

        print(f'Found {len(signal)} electrodes for experiment {experiment}')

        signal = np.stack(signal, axis=-1)  # n_words x n_lags x n_elecs
        pred_signal = np.stack(pred_signal, axis=-1)
        nn_signal = np.stack(nn_signal, axis=-1)
        nnt_signal = np.stack(nnt_signal, axis=-1)
        ynn_test_signal = np.stack(ynn_test_signal, axis=-1)

        corrs, corrs_nn, corrs_nnt, corrs_ynn_test = [], [], [], []
        sems, sems_nn, sems_nnt, sems_ynn_test = [], [], [], []
        rawcorr, rawcorr_nn, rawcorr_nnt, rawcorr_ynn_test = [], [], [], []
        acdw = []

        for lag in range(signal.shape[1]):
            A = signal[:,lag,:]      # n_words x n_elecs
            B = pred_signal[:,lag,:] # n_words x n_elecs
            C = nn_signal[:,lag,:] # n_words x n_elecs
            D = ynn_test_signal[:,lag,:] # n_words x n_elecs
            E = nnt_signal[:,lag,:] # n_words x n_elecs
            # acdw.append(stattools.durbin_watson(A - B, axis=0).mean())
            A = zscore(A, axis=0)
            B = zscore(B, axis=0)
            C = zscore(C, axis=0)
            D = zscore(D, axis=0)
            E = zscore(D, axis=0)

            rs = correlate(A, B, axis=1)  # 1 is rows, 0 is columns
            corrs.append(rs.mean())
            sems.append(rs.std() / np.sqrt(len(rs)))
            rawcorr.append(rs)

            rs = correlate(A, C, axis=1)
            corrs_nn.append(rs.mean())
            sems_nn.append(rs.std() / np.sqrt(len(rs)))
            rawcorr_nn.append(rs)

            rs = correlate(B, D, axis=1)
            corrs_ynn_test.append(rs.mean())
            sems_ynn_test.append(rs.std() / np.sqrt(len(rs)))
            rawcorr_ynn_test.append(rs)

            rs = correlate(A, E, axis=1)
            corrs_nnt.append(rs.mean())
            sems_nnt.append(rs.std() / np.sqrt(len(rs)))
            rawcorr_nnt.append(rs)

        # print('avg DW:', np.mean(acdw))
        nelecs = signal.shape[-1]

        # lags = list(map(str, lags))
        lags = np.asarray(lags) / 1000
        xaxis = lags
        mean = np.asarray(corrs)
        err = np.asarray(sems)
        col = 'blue' if i == 0 else 'gray'
        ax.plot(xaxis, mean, color=col)
        ax.fill_between(xaxis, mean - err, mean + err, alpha=0.1, color=col)

        corrs = np.vstack(rawcorr)
        corrs2 = np.vstack(rawcorr_nn)
        corrs3 = np.vstack(rawcorr_ynn_test)
        corrs4 = np.vstack(rawcorr_nnt)

        df = pd.DataFrame(corrs.T, columns=lags)
        df.insert(0, 'type', 'actual' if i == 0 else 'shuffle')
        dfs.append(df)

        if i == 0: # not the shuffled one
            mean_nn = np.asarray(corrs_nn)
            err_nn = np.asarray(sems_nn)
            ax.plot(xaxis, mean_nn, color='red')
            ax.fill_between(xaxis, mean_nn - err_nn, mean_nn + err_nn, alpha=0.1, color='red')

            df = pd.DataFrame(corrs2.T, columns=lags)
            df.insert(0, 'type', 'near_neighbor')
            dfs.append(df)

            mean_ynn_test = np.asarray(corrs_ynn_test)
            err_ynn_test = np.asarray(sems_ynn_test)
            ax.plot(xaxis, mean_ynn_test, color='green')
            ax.fill_between(xaxis, mean_ynn_test - err_ynn_test, mean_ynn_test + err_ynn_test, alpha=0.1, color='green')

            df = pd.DataFrame(corrs3.T, columns=lags)
            df.insert(0, 'type', 'y_near_neighbor_test')
            dfs.append(df)

            mean_nnt = np.asarray(corrs_nnt)
            err_nnt = np.asarray(sems_nnt)
            ax.plot(xaxis, mean_nnt, color='orange')
            ax.fill_between(xaxis, mean_nnt - err_nnt, mean_nnt + err_nnt, alpha=0.1, color='orange')

            df = pd.DataFrame(corrs4.T, columns=lags)
            df.insert(0, 'type', 'near_neighbor_test')
            dfs.append(df)

        ax.set_title(f'{experiment} | N={nelecs}')

        if i==0 and nperms > 0:
            # Calculate max of correlation significance
            pvals = [one_samp_perm(corrs[i], nperms) for i in range(len(lags))]
            pcorr = fdr(pvals)
            m = corrs.mean(axis=1)
            g = pcorr <= 0.01
            minP = 0
            if g.any():
                sigSig = m[g]
                if (pcorr > 0.01).any():
                    maxP = m[pcorr > 0.01].max()
                    gg = sigSig > maxP
                    if gg.any():
                        minP = sigSig[gg].min()
                        ax.axhline(minP)

            # sig test for nn
            pvals = [paired_permutation(corrs2[i], corrs[i], nperms) for i in range(len(lags))]
            pcorr = fdr(pvals)
            issig = (m > minP) & (pcorr < 0.01)
            siglags = issig.nonzero()[0]
            yheight = ax.get_ylim()[1] - .005
            ax.scatter(lags[siglags], [yheight]*len(siglags), marker='*', color='red')

            dfs[0].insert(1, 'threshold', minP)
            df = pd.DataFrame(issig).T.set_axis(lags, axis=1)
            df.insert(0, 'type', 'sig_nn_train')
            dfs.append(df)

            # sig test for ynnt
            pvals = [paired_permutation(corrs3[i], corrs[i], nperms) for i in range(len(lags))]
            pcorr = fdr(pvals)
            issig = (m > minP) & (pcorr < 0.01)
            siglags = issig.nonzero()[0]
            yheight = ax.get_ylim()[1] - .005
            ax.scatter(lags[siglags], [yheight]*len(siglags), marker='*', color='green')

            df = pd.DataFrame(issig).T.set_axis(lags, axis=1)
            df.insert(0, 'type', 'sig_ynn_test')
            dfs.append(df)

            # sig test for nnt
            pvals = [paired_permutation(corrs4[i], corrs[i], nperms) for i in range(len(lags))]
            pcorr = fdr(pvals)
            issig = (m > minP) & (pcorr < 0.01)
            siglags = issig.nonzero()[0]
            yheight = ax.get_ylim()[1] - .005
            ax.scatter(lags[siglags], [yheight]*len(siglags), marker='*', color='orange')

            df = pd.DataFrame(issig).T.set_axis(lags, axis=1)
            df.insert(0, 'type', 'sig_nn_test')
            dfs.append(df)

        # from matplotlib.backends.backend_pdf import PdfPages
        # pdf = PdfPages('tmp.pdf')
        # fig, ax = plt.subplots()
        # ax.axvline(0, ls='-', c='black', alpha=0.1)
        # ax.axhline(0, ls='-', c='black', alpha=0.1)
        # ax.set(xlabel='Lag (s)', ylabel='Correlation (r+se)')
        # # ax.set_ylim([-0.05, 0.25])
        # ax.plot(lags, corrs.mean(axis=1), color='blue')
        # ax.plot(lags, corrs2.mean(axis=1), color='red')
        # ax.set_title('all')
        # pdf.savefig(fig)
        # plt.close()
        # for j in range(nelecs):
        #     fig, ax = plt.subplots()
        #     ax.axvline(0, ls='-', c='black', alpha=0.1)
        #     ax.axhline(0, ls='-', c='black', alpha=0.1)
        #     ax.set(xlabel='Lag (s)', ylabel='Correlation (r+se)')
        #     # ax.set_ylim([-0.05, 0.25])
        #     ax.plot(lags, corrs[:,j], color='blue')
        #     ax.plot(lags, corrs2[:,j], color='red')
        #     ax.set_title(elecs[j])
        #     pdf.savefig(fig)
        #     plt.close()
        # pdf.close()


    df = pd.concat(dfs)
    df.to_csv(f'results/figures/0shot-dat-{custom}-{experiment}-n_{nelecs}.csv')

    fig.savefig(f'results/figures/0shot-fig-{custom}-{experiment}-n_{nelecs}.png')
    plt.close()


def get_zeroshot_datum(df, is_gpt2, kfolds = 10):

    # Turn into zeroshot
    df['word'] = df.word.str.lower().str.strip(string.punctuation)
    nans = df['embeddings'].isna()
    # nans = df.embeddings.apply(lambda x: np.isnan(x).any())
    notnon = df.is_nonword == 0
    if is_gpt2: # gpt2 datum
        same = df.token2word.str.lower().str.strip() == df.word.str.lower().str.strip()
        df2 = df[same & ~nans & notnon].copy()
    else: # glove datum
        df2 = df[~nans & notnon].copy()
    df2.reset_index(drop=True, inplace=True)
    assert not df2.adjusted_onset.isna().any()
    df3 = df2[['word', 'adjusted_onset']].copy()
    dfz = df3.groupby('word').apply(lambda x: x.sample(1, random_state=42))
    dfz.reset_index(level=1, inplace=True)
    dfz.sort_values('adjusted_onset', inplace=True)
    df = df2.iloc[dfz.level_1.values]

    onsets = df.adjusted_onset.values.astype(int)
    embeddings = np.stack(df.embeddings.values)

    # Shuffled embeddings
    # indices = np.arange(len(embeddings))
    # np.random.shuffle(indices)
    # shuffled_embeddings = embeddings[indices]
    ret_folds_idxs = []
    skf = KFold(n_splits=kfolds, shuffle=False)
    n_samps = len(embeddings)
    folds = [t[1] for t in skf.split(np.arange(n_samps))]
    for i in range(kfolds):
      folds_ixs = np.roll(range(kfolds), i)
      test_fold = folds_ixs[-1]
      train_folds = folds_ixs[:-1]
      test_index = folds[test_fold]
      train_index = np.concatenate([folds[j] for j in train_folds])
      test_embeddings = embeddings[test_index]
      train_embeddings = embeddings[train_index]
      knn=NearestNeighbors(n_neighbors=1, metric='cosine')
      knn.fit(train_embeddings)
      _, I = knn.kneighbors(test_embeddings)
      test_nn_index = train_index[I.squeeze()]
      knn.fit(test_embeddings)
      _, I = knn.kneighbors()
      test_test_nn_index = test_index[I.squeeze()]
      ret_folds_idxs.append((test_index, test_nn_index, train_index, test_test_nn_index))

    return (onsets, ret_folds_idxs)

def get_ynn_test(onsets, folds_idxs):

    ynn_test_idxs = np.zeros(onsets.shape, dtype=int)
    # ynn_idxs = np.zeros(onsets.shape, dtype=int)
    for test_ids, test_nn_ids, _, test_test_nn_ids in folds_idxs:
        ynn_test_idxs[test_ids] = test_test_nn_ids
        # ynn_idxs[test_ids] = test_nn_ids

    return ynn_test_idxs


if __name__ == '__main__':

    pickle_dir = 'data/podcast/777/pickles'
    glove_pickle = '777_full_glove50_layer_01_embeddings.pkl'
    gpt2_pickle = '777_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl'
    kfolds = 10

    file_name = os.path.join(pickle_dir, gpt2_pickle)
    gpt2_emb = True
    if 'glove50' in dirs[0]:
        file_name = os.path.join(pickle_dir, glove_pickle)
        gpt2_emb = False
    df = load_datum(file_name)
    
    onsets, folds_idxs = get_zeroshot_datum(df, gpt2_emb, kfolds)
    ynn_test_idxs = get_ynn_test(onsets, folds_idxs)

    parallel = True
    if parallel:
        with Pool(min(n_workers, len(experiments))) as p:
            p.starmap(partial(run_exp,y_nn_idxs = ynn_test_idxs), experiments.items())
    else:
        for exp_name, exp_elecs in experiments.items():
            run_exp(exp_name, exp_elecs, ynn_test_idxs)