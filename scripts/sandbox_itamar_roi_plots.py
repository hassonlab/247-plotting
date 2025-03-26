"""
Plot encoding results for different ROIs
Plots of interest - average lag/perf averaged per roi
brainmap by roi

Based on ken's code:
/scratch/gpfs/kw1166/247/247-plotting/scripts/tfspaper_sts.ipynb

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches

#%% Electrode selection (before loading data)
# Load electrode names / locations / roi labels
sig_df_whisper = pd.read_csv("/scratch/gpfs/kw1166/247/247-plotting/data/plotting/paper-sts/base_df.csv")
sig_df_whisper.rename(columns={"sid":"subject","elec_1":"electrode"},inplace=True)

# only keep subject 798
sig_df_whisper = sig_df_whisper[sig_df_whisper["subject"] == 798]

# Select subset of significant electrodes
comp_sig_whisper = sig_df_whisper[sig_df_whisper["whisper-en-last-0.01-comp"] | sig_df_whisper["whisper-de-best-0.01-comp"]]
prod_sig_whisper = sig_df_whisper[sig_df_whisper["whisper-en-last-0.01-prod"] | sig_df_whisper["whisper-de-best-0.01-prod"]]

# Pick electrodes for each ROI
comp_sigs = {}
prod_sigs = {}

# rois = ["IFG","STG","SM","TP"]
# rois = ["IFG","STG","TP","dM","vM","mM","dS","vS","mS"]
# rois = ["IFG","STG","TP","dSM","vSM","mSM"]
# rois = ["IFG","STG","preCG","postCG", "All"]
rois = ["IFG","MTG", "STG","preCG","postCG", "rostralmiddlefrontal",
        "superiorfrontal", "supramarginal", "All"]
# rois = ["IFG","STG","preCG","postCG","TP","dSM","mSM","vSM", "All"]


for roi in rois:
    try:
        comp_sigs[roi] = comp_sig_whisper.loc[comp_sig_whisper.roi_1 == roi,("subject","electrode")]
        prod_sigs[roi] = prod_sig_whisper.loc[prod_sig_whisper.roi_1 == roi,("subject","electrode")]
        assert len(comp_sigs[roi]) > 0
        assert len(prod_sigs[roi]) > 0
    except:
        comp_sigs[roi] = comp_sig_whisper.loc[:,("subject","electrode")]
        prod_sigs[roi] = prod_sig_whisper.loc[:,("subject","electrode")]
        assert len(comp_sigs[roi]) > 0
        assert len(prod_sigs[roi]) > 0
    print(f"{roi}, comp {len(comp_sigs[roi])}, prod {len(prod_sigs[roi])}")


#%% Results loading




comp_results = {}
prod_results = {}
data_dir1 = "./data/encoding/tfs/"
tmp_comp = pd.read_csv(f"{data_dir1}/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-2words_comp.csv")
tmp_prod = pd.read_csv(f"{data_dir1}/ij-tfs-%s-gpt2-xl-bandedRidge-lag2k-25-all-static_future_past-2words_prod.csv")


comp_results["enc_joint"] = tmp_comp[tmp_comp["label3"] == "joint"]
comp_results["enc_word"] = tmp_comp[tmp_comp["label3"] == "word"]
comp_results["enc_future"] = tmp_comp[tmp_comp["label3"] == "sentence"]
comp_results["enc_past"] = tmp_comp[tmp_comp["label3"] == "sentence2"] 
prod_results["enc_joint"] = tmp_prod[tmp_prod["label3"] == "joint"]
prod_results["enc_word"] = tmp_prod[tmp_prod["label3"] == "word"]
prod_results["enc_future"] = tmp_prod[tmp_prod["label3"] == "sentence"]
prod_results["enc_past"] = tmp_prod[tmp_prod["label3"] == "sentence2"]


# threshold results
comp_sig_results = {}
prod_sig_results = {}

for result in comp_results.keys():
    for roi in comp_sigs.keys():
        comp_sig_results[(result, roi)] = pd.merge(comp_results[result], comp_sigs[roi], on=["subject", "electrode"])
        prod_sig_results[(result, roi)] = pd.merge(prod_results[result], prod_sigs[roi], on=["subject", "electrode"])

# Combining different lags
LAGS = {
    "2": {
        "lags_all": np.arange(-2000, 2001, 25),
        "lags_plt": np.arange(-2000, 2001, 25),
        "lags": np.arange(-2000, 2001, 25),
        "lag_ticks": np.arange(-2000,2001,500),
        "lag_tick_labels": np.arange(-2, 2.001, 0.5),
    },
    "5": {
        "lags_all": np.arange(-5000, 5001, 25),
        "lags_plt": np.arange(-5000, 5001, 25),
        "lags": np.arange(-5000, 5001, 25),
        "lag_ticks": np.arange(-5000,5001,1000),
        "lag_tick_labels": np.arange(-5, 5.001, 1),
    },
}

# Plotting funcs
class Args(argparse.Namespace):
    # rois = ["IFG"] # roi id
    # rois = ["IFG","STG","TP", "dSM", "mSM", "vSM", "preCG", "postCG"]
    # rois = ["dM","vM","mM","dS","vS","mS"]
    # rois = ["IFG","STG","TP","dSM","vSM","mSM"]
    # rois = ["All"]
    rois = ["IFG","STG","preCG","postCG", "All"]
    rois = ["IFG","MTG", "STG","preCG","postCG", "rostralmiddlefrontal",
        "superiorfrontal", "supramarginal", "All"]
    # rois = ["IFG","STG","preCG","postCG","TP","dSM","mSM","vSM", "All"]
    
    
    lines = ["enc_joint", "enc_word", "enc_future", "enc_past"] # line id
    # colors = ["green", "grey", "orange", "red", "black", "purple"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legends = ["joint", "word", "future", "past"]

    lags = LAGS["2"]
    res_dir = '/scratch/gpfs/ij9216/projects/code/247/247-plotting/results/roi_figures'

def plot_roi(args, df, mode=""):
    mode_full = "Comprehension" if mode == "comp" else "Production"
    for roi in args.rois:
        fig, ax = plt.subplots(figsize=(10, 5))
        

        
        for idx, line in enumerate(args.lines):
            key = (line, roi)
            if key not in df:
                print(f"Warning: Key {key} not found in the data. Skipping...")
                continue
            n_elecs = len(df[key])
            label = f"{args.legends[idx]}"  
            ax = plot_line(ax, args.lags, df[key], args.colors[idx], label)
        
        ymin, ymax = ax.get_ylim() 
        rect1 = patches.Rectangle((-500, ymin), 450, ymax - ymin, color="indianred", alpha=0.3, label="_nolegend_")  # Rectangle 1
        rect2 = patches.Rectangle((50, ymin), 450, ymax - ymin, color="yellowgreen", alpha=0.3, label="_nolegend_")     # Rectangle 2
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_xticks(args.lags["lag_ticks"])
        ax.set_xticklabels(args.lags["lag_tick_labels"])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.legend(loc="best", frameon=False, fontsize=10)  
        plt.title(f"{mode_full} ({roi} - n={n_elecs})", fontsize=14)  
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, top=0.85)
        plt.savefig(f"{args.res_dir}/{roi}_{mode}.jpeg")
    return


def plot_emb(args, df, mode=""):
    for line in args.lines:
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, roi in enumerate(args.rois):
            key = (line, roi)
            if key not in df:
                print(f"Warning: Key {key} not found in the data. Skipping...")
                continue
            n_elecs = len(df[key])
            label = f"{roi} (n={n_elecs})"
            ax = plot_line(ax, args.lags, df[key], args.colors[idx], label)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_xticks(args.lags["lag_ticks"])
        ax.set_xticklabels(args.lags["lag_tick_labels"])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.legend(loc="upper right", frameon=False, fontsize=10) 
        plt.legend(fontsize=10) 
        plt.savefig(f"{args.res_dir}/{line}_{mode}.jpeg")
    return
# def plot_roi(args, df, mode=""):
#     for roi in args.rois:
#         fig, ax = plt.subplots(figsize=(10,5))
#         for idx, line in enumerate(args.lines):
#             ax = plot_line(ax, args.lags, df[(line, roi)], args.colors[idx], args.legends[idx])
#         ax.axhline(0, ls="dashed", alpha=0.3, c="k")
#         ax.axvline(0, ls="dashed", alpha=0.3, c="k")
#         ax.set_xticks(args.lags["lag_ticks"])
#         ax.set_xticklabels(args.lags["lag_tick_labels"])
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         ax.legend(loc="upper right", frameon=False)
#         # ax.set_ylim(-0.02,0.2)
#         plt.legend(fontsize=15)
#         plt.savefig(f"{args.res_dir}/{roi}_{mode}.jpeg")
#     return


# def plot_emb(args, df, mode=""):
#     for line in args.lines:
#         fig, ax = plt.subplots(figsize=(10,5))
#         for idx, roi in enumerate(args.rois):
#             ax = plot_line(ax, args.lags, df[(line, roi)], args.colors[idx], roi)
#             # ax = plot_line(ax, args.lags, df2[(line, roi)], args.colors[idx+1], roi)
#         ax.axhline(0, ls="dashed", alpha=0.3, c="k")
#         ax.axvline(0, ls="dashed", alpha=0.3, c="k")
#         ax.set_xticks(args.lags["lag_ticks"])
#         ax.set_xticklabels(args.lags["lag_tick_labels"])
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         ax.legend(loc="upper right", frameon=False)
#         # ax.set_ylim(-0.02,0.2)
#         plt.legend(fontsize=15)
#         plt.savefig(f"{args.res_dir}/{line}_{mode}.jpeg")
#     return

def plot_line(ax, args, df, color, label):
    lags_selected = [lag_idx for lag_idx, lag in enumerate(args["lags_all"]) if lag in args["lags_plt"]]
    vals = df.iloc[:,lags_selected].mean(axis=0)
    # vals = vals - vals[0]
    errs = df.iloc[:,lags_selected].sem(axis=0)
    ax.plot(
        args["lags"],
        vals,
        color=color,
        label=f"{label}",
        lw=2.5,
    )
    ax.fill_between(
        args["lags"],
        vals - errs,
        vals + errs,
        alpha=0.2,
        color=color,
    )
    return ax



# call plotting funcs
args = Args()
plt.style.use('/scratch/gpfs/kw1166/247/247-plotting/data/plotting/paper-prob-improb/paper.mlpstyle')
plot_roi(args, comp_sig_results, "comp")
plot_roi(args, prod_sig_results, "prod")
plot_emb(args, comp_sig_results, "comp")
plot_emb(args, prod_sig_results, "prod")
print("all done")