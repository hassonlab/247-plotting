import glob
import argparse
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--labels", nargs="+",  required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+",  required=True)
parser.add_argument("--sid", type=int, default=717)
parser.add_argument("--sig-elec-file", nargs="+", default=[])
parser.add_argument("--outfile", default='results/figures/tmp.pdf')
args = parser.parse_args()

assert len(args.labels) == len(args.formats)

elecdir = f'/projects/HASSON/247/data/elecimg/{args.sid}/'

# Assign a unique color/style to each label/mode combination
# i.e. gpt2 will always be blue, prod will always be full line
#      glove will always be red, comp will always be dashed
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
styles = ['-', '--', '-.', ':']
cmap = {}  # color map
smap = {}  # style map
for label, color in zip(args.labels, colors):
    for key, style in zip(args.keys, styles):
        cmap[(label, key)] = color
        smap[(label, key)] = style


def get_elecbrain(electrode):
    name = electrode.replace('EEG', '').replace('REF', '').replace('\\', '')
    name = name.replace('_', '').replace('GR', 'G')
    imname = elecdir + f'thumb_{name}.png'  # + f'{args.sid}_{name}.png'
    return imname


# Read significant electrode file(s)
sigelecs = {}
sigelecs_sorted = []
if len(args.sig_elec_file) == 1 and len(args.keys) > 1:
    for fname, mode in zip(args.sig_elec_file, args.keys):
        elecs = pd.read_csv('data/' + fname % mode)['electrode'].tolist()
        sigelecs[mode] = set(elecs)
if len(args.sig_elec_file) == len(args.keys):
    for fname, mode in zip(args.sig_elec_file, args.keys):
        elecs = pd.read_csv('data/' + fname)['electrode'].tolist()
        sigelecs_sorted.append(elecs)
        sigelecs[mode] = set(elecs)

print('Aggregating data')
data = []
for fmt, label in zip(args.formats, args.labels):
    for key in args.keys:
        fname = fmt % key
        files = glob.glob(fname)
        assert len(files) > 0, f"No results found under {fname}"

        for resultfn in files:
            elec = os.path.basename(resultfn).replace('.csv', '')[:-5]
            # Skip electrodes if they're not part of the sig list
            if len(sigelecs) and elec not in sigelecs[key]:
                continue
            df = pd.read_csv(resultfn, header=None)
            df.insert(0, 'mode', key)
            df.insert(0, 'electrode', elec)
            df.insert(0, 'label', label)
            data.append(df)

if not len(data):
    print('No data found')
    exit(1)
df = pd.concat(data)
df.set_index(['label', 'electrode', 'mode'], inplace=True)
lags = args.values
lags = [lag / 1000 for lag in lags]
n_av, n_df = len(args.values), len(df.columns)
# assert n_av == n_df, \
    # 'args.values length ({n_av}) must be same size as results ({n_df})'

print('Plotting')
pdf = PdfPages(args.outfile)

plot_mode = 'none'
plot_mode = 'final'
plot_mode = 'quardra'
print(f'plotting {plot_mode}')

lag_ticks = lags
if plot_mode == 'none':
    lag_ticks_out = []
elif plot_mode == 'quardra':
    lag_ticks_out = [12,13,14,15,16,18,20,22,24,26,28] # quardra
    lag_tick_locations = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    lag_ticklabels = [-300,-250,-200,-150,-120,-90,-60,-40,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,40,60,90,120,150,200,250,300]
elif plot_mode == 'final':
    lag_ticks_out = [12,14,16] # final
    lag_tick_locations = [-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16]
    lag_ticklabels = [-300,-60,-30,-10,-8,-6,-4,-2,0,2,4,6,8,10,30,60,300]
# lag_ticks_out = [12,13,14,15,16,18,20,22] # triple
for lag in lag_ticks_out:
    lag_ticks.insert(0,lag*-1)
    lag_ticks.append(lag)

# plot_mode = 'none'
# lag_idx = [i for i,lag in enumerate(lag_ticks) if (lag >= -2 and lag <= 2)] # only -2 to 2 s
# df = df[lag_idx]
# lag_ticks = [lag for lag in lag_ticks if (lag >= -2 and lag <= 2)]


# breakpoint()
# Plot results for each key (i.e. average)
fig, ax = plt.subplots(figsize=(15,6))
for mode, subdf in df.groupby(['label', 'mode'], axis=0):
    vals = subdf.mean(axis=0)
    err = subdf.sem(axis=0)
    ax.fill_between(lag_ticks, vals - err, vals + err, alpha=0.2, color=cmap[mode])
    label = '-'.join(mode)
    if plot_mode != 'none':
        ax.set_xticks(lag_tick_locations)
        ax.set_xticklabels(lag_ticklabels)
    ax.plot(lag_ticks, vals, label=f'{label} ({len(subdf)})', color=cmap[mode], ls=smap[mode])
ax.axhline(0,ls='dashed',alpha=0.3,c='k')
ax.axvline(0,ls='dashed',alpha=0.3,c='k')
ax.legend(loc='upper right', frameon=False)
ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title='Global average')
pdf.savefig(fig)
plt.close()

# Plot each electrode separately
# vmax, vmin = df.max().max(), df.min().min()
# for electrode, subdf in df.groupby('electrode', axis=0):
#     fig, ax = plt.subplots()
#     for (label, _, mode), values in subdf.iterrows():
#         mode = (label, mode)
#         label = '-'.join(mode)
#         ax.plot(lags, values, label=label, color=cmap[mode], ls=smap[mode])
#     ax.legend(loc='upper left', frameon=False)
#     ax.set_ylim(vmin - 0.05, vmax + .05)  # .35
#     ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title=f'{electrode}')
#     imname = get_elecbrain(electrode)
#     if os.path.isfile(imname):
#         arr_image = plt.imread(imname, format='png')
#         fig.figimage(arr_image,
#                      fig.bbox.xmax - arr_image.shape[1],
#                      fig.bbox.ymax - arr_image.shape[0], zorder=5)
#     pdf.savefig(fig)
#     plt.close()

# Plot each electrode separately (in the sigfile order)
vmax, vmin = df.max().max(), df.min().min()
for electrode, subdf in df.groupby('electrode', axis=0):
# for single_elecs in sigelecs_sorted[0]: # order by single sig list
    # subdf = df.xs(single_elecs,level='electrode',drop_level=False)
    fig, ax = plt.subplots(figsize=(15,6))
    for (label, _, mode), values in subdf.iterrows():
        mode = (label, mode)
        label = '-'.join(mode)
        ax.plot(lag_ticks, values, label=label, color=cmap[mode], ls=smap[mode])
    ax.legend(loc='upper left', frameon=False)
    ax.axhline(0,ls='dashed',alpha=0.3,c='k')
    ax.axvline(0,ls='dashed',alpha=0.3,c='k')
    if plot_mode != 'none':
        ax.set_xticks(lag_tick_locations)
        ax.set_xticklabels(lag_ticklabels)
    ax.set_ylim(vmin - 0.05, vmax + .05)  # .35
    electrode = values.name[1]
    ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title=f'{electrode}')
    imname = get_elecbrain(electrode)
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format='png')
        fig.figimage(arr_image,
                     fig.bbox.xmax - arr_image.shape[1],
                     fig.bbox.ymax - arr_image.shape[0], zorder=5)
    pdf.savefig(fig)
    plt.close()


pdf.close()