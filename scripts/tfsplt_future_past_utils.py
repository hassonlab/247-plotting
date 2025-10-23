# functions
import os, argparse, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from tfsplt_glassbrain import plot_glassbrain
from tfsplt_brainmap import read_coor





# function definitions

def plot_effect_glassbrain(
    df,
    effect_col,
    subjects=["625", "676", "717", "798"],
    coords_dir="/scratch/gpfs/ij9216/projects/code/247/247-plotting/data/plotting/brainplot/",
    cmap=None,
    outfile="",
    show=True,
    vmin=0,
    vmax=1,
    ax=None,
    title=None,
):


    class Args:
        def __init__(self):
            self.project = "tfs"
            self.main_dir = coords_dir
            # self.effect = "color"
            self.effect = ""
            self.cmap = cmap if cmap is not None else mpl.cm.viridis
            self.outfile = outfile

    args = Args()
    grouped_plot = df.reset_index() if "index" in df.columns or df.index.name is not None else df.copy()
    grouped_plot = grouped_plot.rename(columns={effect_col: "effect"})
    grouped_plot['subject'] = grouped_plot['subject'].astype(str)
    df_coor = read_coor(coords_dir, subjects)
    df_coor.loc[df_coor['subject'] == '717', 'subject'] = '7170'
    df_coor = df_coor.rename(columns={"name": "electrode"})
    grouped_plot = pd.merge(grouped_plot, df_coor, on=["subject", "electrode"], how="left")
    ax = plot_glassbrain(args, grouped_plot, outfile=outfile if not show else "", show=show, vmin=vmin, vmax=vmax, ax=ax)
    if title is not None and ax is not None:
        ax.set_title(title, fontsize=12, pad=10)
    return ax

def plot_roi(args, df, mode="", ymax=None, save=True, plot_indiv=False):
    mode_full = "Comprehension" if mode == "comp" else "Production"
    n_rois = len(args.rois)
    if not save:
        ncols = 2
        nrows = math.ceil(n_rois / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
        axes = axes.flatten()
    for i, roi in enumerate(args.rois):
        if save:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            ax = axes[i]
        for idx, line in enumerate(args.lines):
            key = (line, roi)
            if key not in df:
                print(f"Warning: Key {key} not found in the data. Skipping...")
                continue
            n_elecs = len(df[key])
            label = f"{args.legends[idx]}"
            if plot_indiv == line:
                ax = plot_all_indiv_electrodes(ax, args.lags, df[key], args.colors[idx], label)
            elif plot_indiv==False:
                ax = plot_line(ax, args.lags, df[key], args.colors[idx], label)
            else:
                continue
        if ymax:
            ax.set_ylim(top=ymax, bottom=-0.025)
        ymin, ymax_val = ax.get_ylim()
        if mode == "comp":
            rect1 = patches.Rectangle((50, ymin), 450, ymax_val - ymin, color="yellowgreen", alpha=0.3, label="_nolegend_")
        elif mode == "prod":
            rect1 = patches.Rectangle((-500, ymin), 450, ymax_val - ymin, color="indianred", alpha=0.3, label="_nolegend_")
        ax.add_patch(rect1)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_xticks(args.lags["lag_ticks"])
        ax.set_xticklabels(args.lags["lag_tick_labels"])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.legend(loc="best", frameon=False, fontsize=10)
        ax.set_title(f"{mode_full} ({roi} - n={n_elecs})", fontsize=14)
        if save:
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.85)
            plt.savefig(f"{args.res_dir}/{roi}_{mode}.jpeg")
            plt.close(fig)
    if not save:
        plt.tight_layout()
        plt.show()
    return

def plot_line(ax, args, df, color, label):
    lags_selected = [lag_idx for lag_idx, lag in enumerate(args["lags_all"]) if lag in args["lags_plt"]]
    vals = df.iloc[:,lags_selected].mean(axis=0)
    # vals = vals - vals[0]
    errs = df.iloc[:,lags_selected].sem(axis=0)
    ax.plot(
        # args["lags"],
        args["lags_plt"],
        vals,
        color=color,
        label=f"{label}",
        lw=2.5,
    )
    ax.fill_between(
        # args["lags"],
        args["lags_plt"],
        vals - errs,
        vals + errs,
        alpha=0.2,
        color=color,
    )
    return ax

def plot_all_indiv_electrodes(ax, args, df, color, label):
    lags_selected = [lag_idx for lag_idx, lag in enumerate(args["lags_all"]) if lag in args["lags_plt"]]
    vals = df.iloc[:, lags_selected].mean(axis=0)
    errs = df.iloc[:, lags_selected].sem(axis=0)
    indiv_vals = df.iloc[:, lags_selected]
    # Use a colormap for individual lines
    cmap = plt.get_cmap('tab20')
    n_lines = indiv_vals.shape[0]
    for i in range(n_lines):
        ax.plot(
            args["lags_plt"],
            indiv_vals.iloc[i, :],
            color=cmap(i % 20),  # cycle through 20 colors
            alpha=0.4,
            lw=1,
        )
    # plot mean line
    ax.plot(
        args["lags_plt"],
        vals,
        color=color,
        label=f"{label}",
        lw=2.5,
    )
    # plot error band
    ax.fill_between(
        args["lags_plt"],
        vals - errs,
        vals + errs,
        alpha=0.2,
        color=color,
    )
    return ax

def add_roi_label_to_results(df):
    all_elecs = pd.read_csv("/scratch/gpfs/HASSON/kw1166/247/247-plotting/data/plotting/paper-sts/base_df.csv")
    roi_lookup = all_elecs.set_index(['subject', 'electrode'])['roi_1'].to_dict()
    # Assign ROI to banded_comp
    df['roi'] = df.apply(
        lambda row: roi_lookup.get((row['subject'], row['electrode']), None), axis=1
    )
    return df


def threshold_results_by_joint(df, thresh):
    joint_rows = df[df['label3'] == 'joint']
    max_mask = joint_rows.iloc[:, :-5].max(axis=1) > thresh
    subject_electrode_list = [f"{row['subject']}_{row['electrode']}" for _, row in joint_rows[max_mask].iterrows()]
    # only keep rows where subject_electrode is in the list
    df['subject_electrode'] = df['subject'].astype(str) + '_' + df['electrode'].astype(str)
    df = df[df['subject_electrode'].isin(subject_electrode_list)]
    df = df.drop(columns=['subject_electrode'])
    return df

def _plot_roi_on_ax(ax, args, sig_results, roi, mode="comp", ymax=None, plot_indiv=False, title_prefix="", df_i=None):
    """
    Plot one ROI onto a provided Matplotlib axis from a sig_results dict.
    sig_results: dict keyed by (line, roi) -> DataFrame
    """
    n_elecs = 0
    elec_set = set()

    key_size = len(list(sig_results.keys())[0])

    for idx, line in enumerate(args.lines):
        key = (line, roi) if key_size == 2 else (df_i, line, roi)
        if key not in sig_results or sig_results[key].empty:
            continue
        df_line = sig_results[key]
        # count electrodes across lines
        if "electrode" in df_line.columns:
            elec_set.update(df_line["electrode"].astype(str).tolist())
        n_elecs = max(n_elecs, len(df_line))

        label = f"{args.legends[idx]}"
        if plot_indiv == line:
            _ = plot_all_indiv_electrodes(ax, args.lags, df_line, args.colors[idx], label)
        else:
            _ = plot_line(ax, args.lags, df_line, args.colors[idx], label)

    # If nothing was plotted, indicate no data
    if n_elecs == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return

    # Ax cosmetics
    if ymax is not None:
        ax.set_ylim(top=ymax, bottom=-0.025)
    ymin, ymax_val = ax.get_ylim()

    # Shade time window (match original behavior)
    if mode == "comp":
        rect1 = patches.Rectangle((50, ymin), 450, ymax_val - ymin, color="yellowgreen", alpha=0.3, label="_nolegend_")
    elif mode == "prod":
        rect1 = patches.Rectangle((-500, ymin), 450, ymax_val - ymin, color="indianred", alpha=0.3, label="_nolegend_")
    ax.add_patch(rect1)

    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.set_xticks(args.lags["lag_ticks"])
    ax.set_xticklabels(args.lags["lag_tick_labels"])
    ax.tick_params(axis='both', which='both', labelsize=10)
    ax.legend(loc="best", frameon=False, fontsize=10)
    mode_full = "Comprehension" if mode == "comp" else "Production"
    ax.set_title(f"{title_prefix} {mode_full} ({roi} - n={len(elec_set) if elec_set else n_elecs})", fontsize=10)

def load_res_add_roi_threshold(f, thresh):
    """
    Load results from CSV, add ROI labels, and threshold by joint significance.
    
    Parameters:
    -----------
    f : str
        File path to the CSV results
    thresh : float
        Threshold for joint significance
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with ROI labels and threshold applied
    """
    df = pd.read_csv(f)
    df = add_roi_label_to_results(df)
    if thresh is not None:
        df = threshold_results_by_joint(df, thresh)
    return df


def plot_single_roi_side_by_side(args, sig_results_dict, roi, mode="comp", ymax=0.18, save=False, titles=None):
    """
    Plot a single ROI side-by-side for multiple datasets.
    
    Parameters:
    -----------
    args : Args object
        Contains configuration parameters
    sig_results_list : list of dict
        List of sig_results dictionaries, each keyed by (line, roi)
    roi : str
        ROI name to plot
    mode : str
        "comp" or "prod"
    ymax : float
        Maximum y-axis value
    save : bool
        Whether to save the figure
    titles : list of str, optional
        Custom titles for each subplot. If None, uses generic titles.
    """

    # key = (line, roi)
    #     if key not in sig_results or sig_results[key].empty:
    #         continue
    #     df_line = sig_results[key]
    #     # count electrodes across lines
    #     if "electrode" in df_line.columns:
    #         elec_set.update(df_line["electrode"].astype(str).tolist())
    #     n_elecs = max(n_elecs, len(df_line))

    #     label = f"{args.legends[idx]}"
    #     if plot_indiv == line:
    #         _ = plot_all_indiv_electrodes(ax, args.lags, df_line, args.colors[idx], label)
    #     else:
    #         _ = plot_line(ax, args.lags, df_line, args.colors[idx], label)


    # get all keys from dict

    n_plots = len(titles)
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    
    # Handle case where there's only one plot (axes is not a list)
    if n_plots == 1:
        axes = [axes]
    
    # Default titles if not provided
    if titles is None:
        titles = [f"Dataset {i+1} Results -" for i in range(n_plots)]


    for i, (ax, title) in enumerate(zip(axes, titles)):
        _plot_roi_on_ax(ax, args, sig_results_dict, roi, mode=mode, ymax=ymax, plot_indiv=False, title_prefix=title, df_i=i)

    plt.tight_layout()
    if save:
        plt.savefig(f"{args.res_dir}/{roi}_side_by_side_{mode}.jpeg")
        plt.close(fig)
    else:
        plt.show()

def select_shared_elecs_logical_or(dfs, thresh):
    elecs = []
    for i, df in enumerate(dfs):
        tmp_df = threshold_results_by_joint(df, thresh)
        elecs.append(set(tmp_df['electrode']))
    shared_electrodes = set().union(*elecs)
    return shared_electrodes

def process_shared_electrodes_roi(dfs, rois, results=['joint', 'word', 'sentence', 'sentence2'], OR_thresh=None):
    """
    Process a list of dataframes to find shared electrodes and filter by ROI and result type.
    
    Parameters:
    -----------
    dfs : list of pd.DataFrame
        List of dataframes, each containing 'electrode', 'roi', and 'label3' columns
    rois : list of str
        List of ROI names to filter by
    results : list of str, optional
        List of result types (label3 values) to filter by
        
    Returns:
    --------
    tuple : (shared_electrodes, filtered_dfs, roi_results)
        - shared_electrodes: set of electrode IDs present in all dataframes
        - filtered_dfs: list of dataframes filtered to shared electrodes only
        - roi_results: dict with keys (result, roi) containing filtered dataframes
    """
    # Find shared electrodes across all dataframes
    if OR_thresh is not None:
        shared_electrodes = select_shared_elecs_logical_or(dfs, OR_thresh)
    else:
        shared_electrodes = set(dfs[0]['electrode'])
        for df in dfs[1:]:
            shared_electrodes = shared_electrodes.intersection(set(df['electrode']))

    # Filter each dataframe to keep only shared electrodes
    filtered_dfs = [df[df['electrode'].isin(shared_electrodes)].copy() for df in dfs]
    
    # Create ROI-filtered results dictionary
    roi_results = {}
    for i, df in enumerate(filtered_dfs):
        for result in results:
            for roi in rois:
                key = (i, result, roi)  # Include df index to distinguish between datasets
                roi_results[key] = df[(df['label3'] == result) & (df['roi'] == roi)]
    
    return shared_electrodes, filtered_dfs, roi_results

def get_shared_indices(lags_large, lags_small):
    """
    Get indices of shared lag values between two lag arrays.
    
    Parameters:
    -----------
    lags_large : np.ndarray
        Larger array of lag values
    lags_small : np.ndarray
        Smaller array of lag values
        
    Returns:
    --------
    tuple : (shared_indices_large, shared_indices_small)
        - shared_indices_large: indices in lags_large that are also in lags_small
        - shared_indices_small: indices in lags_small that are also in lags_large
    """
    shared_indices_large = np.nonzero(np.isin(lags_large, lags_small))[0]
    shared_indices_small = np.nonzero(np.isin(lags_small, lags_large))[0]
    return shared_indices_large, shared_indices_small

def select_shared_and_last_columns(df, shared_indices, last_n=5):
    """
    Select shared column indices and the last N columns from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    shared_indices : list or array
        Indices of shared columns to select.
    last_n : int
        Number of columns to select from the end.

    Returns:
    --------
    pd.DataFrame
        DataFrame with selected columns.
    """
    shared_columns = df.iloc[:, shared_indices]
    last_columns = df.iloc[:, -last_n:]
    return pd.concat([shared_columns, last_columns], axis=1)
