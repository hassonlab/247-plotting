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
import inspect





# function definitions

def plot_effect_glassbrain(
    df,
    effect_col,
    subjects=["625", "676", "717", "798"],
    coords_dir="/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/data/plotting/brainplot/",
    cmap=None,
    outfile="",
    show=True,
    vmin=0,
    vmax=1,
    ax=None,
    title=None,
    colorbar=True,
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

    # Backward/forward compatibility: some plot_glassbrain versions don't take `colorbar`.
    kwargs = {
        "outfile": outfile if not show else "",
        "show": show,
        "vmin": vmin,
        "vmax": vmax,
        "ax": ax,
    }
    try:
        sig = inspect.signature(plot_glassbrain)
        if "colorbar" in sig.parameters:
            kwargs["colorbar"] = colorbar
    except (TypeError, ValueError):
        # If signature introspection fails, call without `colorbar`.
        pass

    ax = plot_glassbrain(args, grouped_plot, **kwargs)
    if title is not None and ax is not None:
        ax.set_title(title, fontsize=12, pad=10)
    return ax

def filter_valid_rois(df, lines, rois):
    """
    Filter ROIs to only those that have data for at least one line.
    
    Parameters:
    -----------
    df : dict
        Dictionary keyed by (line, roi) tuples containing DataFrames
    lines : list
        List of line names to check
    rois : list
        List of ROI names to filter
        
    Returns:
    --------
    list
        Filtered list of ROIs that have data for at least one line
    """
    valid_rois = []
    for roi in rois:
        # Check if this ROI has data for at least one line
        has_data = any((line, roi) in df and not df[(line, roi)].empty for line in lines)
        if has_data:
            valid_rois.append(roi)
        # else:
        #     print(f"  Skipping ROI '{roi}' - no data for any line")
    
    return valid_rois


def plot_roi(args, df, mode="", ymax=None, save=True, plot_indiv=False):
    mode_full = "Comprehension" if mode == "comp" else "Production"
    
    # Filter to only valid ROIs that have data
    valid_rois = filter_valid_rois(df, args.lines, args.rois)
    n_rois = len(valid_rois)
    
    if n_rois == 0:
        print(f"Warning: No valid ROIs found with data. Skipping plot.")
        return
    
    # print(f"Plotting {n_rois} ROIs (filtered from {len(args.rois)} total)")
    
    if not save:
        ncols = 2
        nrows = math.ceil(n_rois / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
        axes = axes.flatten()
    
    for i, roi in enumerate(valid_rois):
        if save:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            ax = axes[i]
        for idx, line in enumerate(args.lines):
            key = (line, roi)
            if key not in df:
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
        ax.set_xlim(min(args.lags["lags_plt"]), max(args.lags["lags_plt"]))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.legend(loc="best", frameon=False, fontsize=10)
        ax.set_title(f"{mode_full} ({roi} - n={n_elecs})", fontsize=14)
        if save:
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.85)
            if save == 'svg':
                plt.savefig(f"{args.res_dir}/{roi}_{mode}.svg")
            else:
                plt.savefig(f"{args.res_dir}/{roi}_{mode}.jpeg")
            plt.close(fig)
    if not save:
        plt.tight_layout()
        plt.show()
    return

def plot_roi_sep_context(args, df, mode="", ymax=None, save=True, plot_indiv=False, context_thresh_f=0, context_thresh_p=0):
    """
    Plot ROIs with separate context coloring.
    Future (sentence) after context_thresh_f and Past (sentence2) before context_thresh_p are plotted in 'dimgrey'.
    
    Parameters:
    -----------
    args : Args object
        Contains configuration parameters
    df : dict
        Dictionary keyed by (line, roi) tuples containing DataFrames
    mode : str
        "comp" or "prod"
    ymax : float
        Maximum y-axis value
    save : bool
        Whether to save individual figures
    plot_indiv : bool or str
        Whether to plot individual electrodes
    context_thresh : int
        Lag threshold in milliseconds to separate context from target.
        Future after this threshold is context (grey).
        Past before this threshold is context (grey).
    """
    mode_full = "Comprehension" if mode == "comp" else "Production"
    
    # Filter to only valid ROIs that have data
    valid_rois = filter_valid_rois(df, args.lines, args.rois)
    n_rois = len(valid_rois)
    
    if n_rois == 0:
        print(f"Warning: No valid ROIs found with data. Skipping plot.")
        return
    
    if not save:
        ncols = 2
        nrows = math.ceil(n_rois / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
        axes = axes.flatten()
    
    for i, roi in enumerate(valid_rois):
        if save:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            ax = axes[i]
        
        for idx, line in enumerate(args.lines):
            key = (line, roi)
            if key not in df:
                continue
            n_elecs = len(df[key])
            
            # Determine if this line should be split into target and context
            if line == 'sentence':
                # Future: target before/at threshold, context after
                label_target = f"{args.legends[idx]} (target)"
                label_context = f"{args.legends[idx]} (context)"
                ax = plot_line_sep_context(
                    ax, args.lags, df[key], args.colors[idx], 'dimgrey',
                    label_target, label_context, context_thresh_f, split_after=True
                )
            elif line == 'sentence2':
                # Past: context before threshold, target after/at
                label_target = f"{args.legends[idx]} (target)"
                label_context = f"{args.legends[idx]} (context)"
                ax = plot_line_sep_context(
                    ax, args.lags, df[key], args.colors[idx], 'dimgrey',
                    label_target, label_context, context_thresh_p, split_after=False
                )
            else:
                # Word and joint: no split, use original color
                label = f"{args.legends[idx]}"
                if plot_indiv == line:
                    ax = plot_all_indiv_electrodes(ax, args.lags, df[key], args.colors[idx], label)
                elif plot_indiv == False:
                    ax = plot_line(ax, args.lags, df[key], args.colors[idx], label)
                else:
                    continue
        
        if ymax:
            ax.set_ylim(top=ymax, bottom=-0.025)
        ymin, ymax_val = ax.get_ylim()
        # if mode == "comp":
        #     rect1 = patches.Rectangle((50, ymin), 450, ymax_val - ymin, color="yellowgreen", alpha=0.3, label="_nolegend_")
        # elif mode == "prod":
        #     rect1 = patches.Rectangle((-500, ymin), 450, ymax_val - ymin, color="indianred", alpha=0.3, label="_nolegend_")
        # ax.add_patch(rect1)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(context_thresh_f, ls="dotted", alpha=0.5, c="purple", lw=2)  # Mark context threshold
        ax.set_xticks(args.lags["lag_ticks"])
        ax.set_xticklabels(args.lags["lag_tick_labels"])
        ax.set_xlim(min(args.lags["lags_plt"]), max(args.lags["lags_plt"]))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.legend(loc="best", frameon=False, fontsize=10)
        ax.set_title(f"{mode_full} ({roi} - n={n_elecs})", fontsize=14)
        if save:
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.85)
            if save == 'svg':
                plt.savefig(f"{args.res_dir}/{roi}_{mode}_sep_context_{context_thresh_f}.svg")
            else:
                plt.savefig(f"{args.res_dir}/{roi}_{mode}_sep_context_{context_thresh_f}.jpeg")
            plt.close(fig)
    if not save:
        plt.tight_layout()
        plt.show()
    return

def plot_line_sep_context(ax, args, df, color_target, color_context, label_target, label_context, 
                          context_thresh, split_after=True):
    """
    Plot a line with separate colors for target and context regions.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    args : dict
        Dictionary with 'lags_all' and 'lags_plt' keys
    df : DataFrame
        Data to plot
    color_target : str
        Color for target region
    color_context : str
        Color for context region
    label_target : str
        Label for target region
    label_context : str
        Label for context region
    context_thresh : int
        Lag threshold in milliseconds
    split_after : bool
        If True: target <= thresh, context > thresh (for future/sentence)
        If False: context < thresh, target >= thresh (for past/sentence2)
    """
    # Select lag indices that correspond to lags_plt
    lags_selected_indices = []
    lags_selected_values = []
    for lag_val in args["lags_plt"]:
        if lag_val in args["lags_all"]:
            idx = np.where(np.array(args["lags_all"]) == lag_val)[0][0]
            lags_selected_indices.append(str(idx))
            lags_selected_values.append(lag_val)
    
    if not lags_selected_indices:
        print(f"Warning: No matching lag columns found for plotting")
        return ax
    
    vals = df[lags_selected_indices].mean(axis=0).values
    errs = df[lags_selected_indices].sem(axis=0).values
    lags_array = np.array(lags_selected_values)
    
    # Split into target and context regions
    if split_after:
        # Target: lags <= threshold, Context: lags >= threshold (overlap at threshold for continuity)
        target_mask = lags_array <= context_thresh
        context_mask = lags_array >= context_thresh
    else:
        # Context: lags <= threshold, Target: lags >= threshold (overlap at threshold for continuity)
        context_mask = lags_array <= context_thresh
        target_mask = lags_array >= context_thresh
    
    # Plot target region
    if np.any(target_mask):
        ax.plot(
            lags_array[target_mask],
            vals[target_mask],
            color=color_target,
            label=label_target,
            lw=2.5,
        )
        ax.fill_between(
            lags_array[target_mask],
            vals[target_mask] - errs[target_mask],
            vals[target_mask] + errs[target_mask],
            alpha=0.2,
            color=color_target,
        )
    
    # Plot context region
    if np.any(context_mask):
        ax.plot(
            lags_array[context_mask],
            vals[context_mask],
            color=color_context,
            label=label_context,
            lw=2.5,
            linestyle='--',  # Dashed line for context
        )
        ax.fill_between(
            lags_array[context_mask],
            vals[context_mask] - errs[context_mask],
            vals[context_mask] + errs[context_mask],
            alpha=0.2,
            color=color_context,
        )
    
    return ax

def plot_line(ax, args, df, color, label):
    desired_lags = list(args["lags_plt"])
    lags_all = list(args["lags_all"])

    def _select_lag_columns(_df: pd.DataFrame) -> pd.DataFrame:
        # 1) Columns are lag values directly (common case)
        desired_as_str = [str(l) for l in desired_lags]
        if all(c in _df.columns for c in desired_as_str):
            return _df[desired_as_str]
        if all(l in _df.columns for l in desired_lags):
            return _df[desired_lags]

        # 2) Columns are coercible to ints matching lag values
        col_to_int = {}
        for c in _df.columns:
            try:
                col_to_int[c] = int(c)
            except Exception:
                continue
        if col_to_int:
            needed = set(desired_lags)
            have = {v for v in col_to_int.values()}
            if needed.issubset(have):
                inv = {v: k for k, v in col_to_int.items()}
                return _df[[inv[l] for l in desired_lags]]

        # 3) Fallback: many of our result DataFrames store lag columns as numeric
        # (often renamed to original column indices like '560'..'640') plus trailing
        # metadata columns (subject/electrode/roi/etc). Prefer numeric-only columns.
        numeric_df = _df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            n_lags_all = len(lags_all)
            if numeric_df.shape[1] == n_lags_all and n_lags_all > 0:
                idxs = [lags_all.index(l) for l in desired_lags if l in lags_all]
                if len(idxs) == len(desired_lags):
                    return numeric_df.iloc[:, idxs]
            if numeric_df.shape[1] >= len(desired_lags):
                return numeric_df.iloc[:, : len(desired_lags)]

        raise KeyError("No matching lag columns found for plotting")

    try:
        df_sel = _select_lag_columns(df)
    except KeyError:
        print("Warning: No matching lag columns found for plotting")
        return ax

    vals = df_sel.mean(axis=0)
    # vals = vals - vals[0]
    errs = df_sel.sem(axis=0)
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
    desired_lags = list(args["lags_plt"])
    lags_all = list(args["lags_all"])

    def _select_lag_columns(_df: pd.DataFrame) -> pd.DataFrame:
        desired_as_str = [str(l) for l in desired_lags]
        if all(c in _df.columns for c in desired_as_str):
            return _df[desired_as_str]
        if all(l in _df.columns for l in desired_lags):
            return _df[desired_lags]

        col_to_int = {}
        for c in _df.columns:
            try:
                col_to_int[c] = int(c)
            except Exception:
                continue
        if col_to_int:
            needed = set(desired_lags)
            have = {v for v in col_to_int.values()}
            if needed.issubset(have):
                inv = {v: k for k, v in col_to_int.items()}
                return _df[[inv[l] for l in desired_lags]]

        numeric_df = _df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            n_lags_all = len(lags_all)
            if numeric_df.shape[1] == n_lags_all and n_lags_all > 0:
                idxs = [lags_all.index(l) for l in desired_lags if l in lags_all]
                if len(idxs) == len(desired_lags):
                    return numeric_df.iloc[:, idxs]
            if numeric_df.shape[1] >= len(desired_lags):
                return numeric_df.iloc[:, : len(desired_lags)]

        raise KeyError("No matching lag columns found for plotting")

    try:
        indiv_vals = _select_lag_columns(df)
    except KeyError:
        print("Warning: No matching lag columns found for plotting")
        return ax

    vals = indiv_vals.mean(axis=0)
    errs = indiv_vals.sem(axis=0)
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
    joint_rows = df[df["label3"] == "joint"]
    if len(joint_rows) == 0:
        print("Warning: threshold_results_by_joint: no 'joint' rows found; returning unthresholded df")
        return df

    # Use explicit lag columns rather than assuming a fixed number of trailing metadata columns.
    lag_pairs = _get_numeric_lag_columns(joint_rows)
    lag_cols = [c for _, c in lag_pairs]
    if not lag_cols:
        print("Warning: threshold_results_by_joint: no lag-like columns found; returning unthresholded df")
        return df

    max_mask = joint_rows.loc[:, lag_cols].max(axis=1) > thresh
    subject_electrode_list = [
        f"{row['subject']}_{row['electrode']}" for _, row in joint_rows.loc[max_mask].iterrows()
    ]

    # only keep rows where subject_electrode is in the list
    df = df.copy()
    df["subject_electrode"] = df["subject"].astype(str) + "_" + df["electrode"].astype(str)
    df = df[df["subject_electrode"].isin(subject_electrode_list)]
    df = df.drop(columns=["subject_electrode"])
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

    # Normalize label3 across result sources.
    # Many aggregated CSVs store band labels as "_banded_joint" etc.
    if "label3" in df.columns:
        try:
            df["label3"] = df["label3"].astype(str).str.replace("_banded_", "", regex=False)
        except Exception:
            pass

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

def get_shared_indices(lags_large, lags_small, lags_med=None):
    """
    Get indices of shared lag values between two or three lag arrays.
    
    Parameters:
    -----------
    lags_large : np.ndarray
        Larger array of lag values
    lags_small : np.ndarray
        Smaller array of lag values
    lags_med : np.ndarray, optional
        Medium array of lag values (if provided, finds shared values across all three)
        
    Returns:
    --------
    tuple : If lags_med is None:
        (shared_indices_large, shared_indices_small)
        - shared_indices_large: indices in lags_large that are also in lags_small
        - shared_indices_small: indices in lags_small that are also in lags_large
        
    tuple : If lags_med is provided:
        (shared_indices_large, shared_indices_med, shared_indices_small)
        - shared_indices_large: indices in lags_large that are shared across all three
        - shared_indices_med: indices in lags_med that are shared across all three
        - shared_indices_small: indices in lags_small that are shared across all three
    """
    if lags_med is None:
        # Original two-array behavior
        shared_indices_large = np.nonzero(np.isin(lags_large, lags_small))[0]
        shared_indices_small = np.nonzero(np.isin(lags_small, lags_large))[0]
        return shared_indices_large, shared_indices_small
    else:
        # Three-array behavior: find values shared across all three
        # Find intersection of all three arrays
        shared_values = np.intersect1d(lags_large, lags_small)
        shared_values = np.intersect1d(shared_values, lags_med)
        
        # Get indices in each array
        shared_indices_large = np.nonzero(np.isin(lags_large, shared_values))[0]
        shared_indices_med = np.nonzero(np.isin(lags_med, shared_values))[0]
        shared_indices_small = np.nonzero(np.isin(lags_small, shared_values))[0]
        
        return shared_indices_large, shared_indices_med, shared_indices_small

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


def _get_numeric_lag_columns(df):
    """Return (lag_value, column_name) pairs for columns that look like lag columns."""
    lag_pairs = []
    for c in df.columns:
        # common metadata columns in these result CSVs
        if c in {"subject", "electrode", "roi", "label", "label2", "label3", "threshold", "fold", "label1", "label2"}:
            continue
        try:
            # lag columns are typically ints encoded as strings
            lag_pairs.append((int(float(c)), c))
        except Exception:
            continue
    lag_pairs.sort(key=lambda t: t[0])
    return lag_pairs


def select_lag_range_and_last_columns(df, lag_min=-2000, lag_max=2000, last_n=5):
    """Select a lag window plus trailing metadata columns.

    IMPORTANT: Many of our aggregated result CSVs store lag columns as *index strings*
    ("0", "1", ..., "N") rather than actual lag values. In that case, we infer the
    underlying symmetric lag grid and select the indices corresponding to the requested
    [lag_min, lag_max] window.

    For result tables whose lag columns are actual lag values (e.g. "-2000", "-1950", ...)
    we fall back to selecting by numeric lag.
    """

    if last_n < 0:
        raise ValueError("last_n must be >= 0")

    # Case A: lag columns are stored as index strings "0".."N" (common in this codebase).
    numeric_like_cols = [c for c in df.columns if str(c).lstrip("-").isdigit()]
    if numeric_like_cols:
        try:
            idxs = sorted(int(c) for c in numeric_like_cols)
        except Exception:
            idxs = []

        # Require contiguous 0..N to avoid misclassifying numeric metadata columns.
        if idxs and idxs[0] == 0 and idxs == list(range(idxs[-1] + 1)):
            n_lag_cols = idxs[-1] + 1

            # Heuristic: most runs use 50ms step; infer from requested window if possible.
            # (If window is a multiple of 50ms, assume 50; otherwise fall back to 10.)
            step = 50 if (lag_max - lag_min) % 50 == 0 else 10

            half_range = int(((n_lag_cols - 1) // 2) * step)
            source_lags = np.arange(-half_range, half_range + step, step)
            target_lags = np.arange(lag_min, lag_max + step, step)

            # Map target lag values to index columns.
            start = source_lags[0]
            selected_cols = []
            for lag in target_lags:
                if lag < source_lags[0] or lag > source_lags[-1]:
                    continue
                col_idx = int((lag - start) / step)
                col_name = str(col_idx)
                if col_name in df.columns:
                    selected_cols.append(col_name)

            selected_lags = df.loc[:, selected_cols]
            last_columns = df.iloc[:, -last_n:] if last_n else df.iloc[:, 0:0]
            out = pd.concat([selected_lags, last_columns], axis=1)
            out = out.loc[:, ~out.columns.duplicated()]
            return out

    # Case B: lag columns are actual lag values (as strings or numbers).
    lag_pairs = _get_numeric_lag_columns(df)
    lag_cols = [c for lag, c in lag_pairs if lag_min <= lag <= lag_max]
    selected_lags = df.loc[:, lag_cols]
    last_columns = df.iloc[:, -last_n:] if last_n else df.iloc[:, 0:0]
    out = pd.concat([selected_lags, last_columns], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def calculate_auc(df, label3_val, lag_start, lag_end):
    """
    Calculate AUC for electrodes with a specific label3 value within a lag window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with lag columns and label3
    label3_val : str
        'sentence', 'sentence2', 'joint', or 'word'
    lag_start : int
        Start of lag window (ms)
    lag_end : int
        End of lag window (ms)
    
    Returns:
    --------
    pd.DataFrame with electrode, subject, roi, auc columns
    """
    from scipy import integrate
    
    # Filter by label3
    df_filt = df[df['label3'] == label3_val].copy()
    
    # Get lag columns within window (columns can be strings, integers, or floats)
    lag_cols = []
    lag_vals = []
    for col in df_filt.columns:
        # Check if column is numeric (int, float, or numpy numeric types)
        # numpy types need special handling - convert to python types for comparison
        if isinstance(col, (np.integer, np.floating)):
            col_val = float(col)
            if lag_start <= col_val <= lag_end:
                lag_cols.append(col)
                lag_vals.append(col_val)
        elif isinstance(col, (int, float)):
            col_val = float(col)
            if lag_start <= col_val <= lag_end:
                lag_cols.append(col)
                lag_vals.append(col_val)
        elif isinstance(col, str):
            # Try to convert string to numeric
            try:
                col_val = float(col)
                if lag_start <= col_val <= lag_end:
                    lag_cols.append(col)
                    lag_vals.append(col_val)
            except (ValueError, TypeError):
                continue
    
    if len(lag_cols) == 0:
        print(f"Warning: No lag columns found in window [{lag_start}, {lag_end}]")
        return pd.DataFrame(columns=['subject', 'electrode', 'roi', 'label3', 'lag_start', 'lag_end', 'auc', 'n_lags'])
    
    # Sort by numeric values
    sorted_indices = np.argsort(lag_vals)
    lag_cols_sorted = [lag_cols[i] for i in sorted_indices]
    lag_vals_sorted = [lag_vals[i] for i in sorted_indices]
    
    results = []
    
    for idx, row in df_filt.iterrows():
        # Get values in time window
        vals = row[lag_cols_sorted].values
        
        # Calculate AUC using trapezoidal rule
        # x values are the numeric lag values, y values are the correlations
        auc = integrate.trapezoid(vals, x=lag_vals_sorted)
        
        results.append({
            'subject': row['subject'],
            'electrode': row['electrode'],
            'roi': row.get('roi', 'unknown'),
            'label3': label3_val,
            'lag_start': lag_start,
            'lag_end': lag_end,
            'auc': auc,
            'n_lags': len(lag_cols_sorted)
        })
    
    return pd.DataFrame(results)


def get_word_peak_share(df, thresh=0.1):
    """
    Calculate the ratio of word peak to joint peak for each electrode.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with label3 column and lag columns
    thresh : float
        Threshold for joint performance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with electrode, subject, word_peak_share columns
    """
    # Get numeric lag columns
    lag_cols = [col for col in df.columns if isinstance(col, (int, float, np.integer, np.floating))]
    
    df['max'] = df[lag_cols].max(axis=1)
    # make a df grouped by subject and electrode with max_{label3} columns
    grouped = df.groupby(['subject', 'electrode', 'label3'])['max'].max().unstack().reset_index()
    # make a new column with the ratio of word peak / joint peak
    grouped = grouped[grouped['joint'] > thresh]
    grouped['word_peak_share'] = grouped['word'] / (grouped['joint'])
    return grouped


def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function with numerical stability.
    
    Parameters:
    -----------
    x : array
        Independent variable (e.g., time lags)
    L : float
        Maximum value (upper asymptote)
    x0 : float
        Midpoint (inflection point)
    k : float
        Steepness/growth rate (positive=increasing, negative=decreasing)
    b : float
        Baseline offset (lower asymptote)
    """
    # Clip the exponent to avoid overflow
    exponent = -k * (x - x0)
    exponent = np.clip(exponent, -500, 500)  # Prevent overflow
    return L / (1 + np.exp(exponent)) + b


def fit_sigmoid_to_row(row_data, x_values, force_direction=None):
    """
    Fit sigmoid to a single row of data.
    
    Parameters:
    -----------
    row_data : array
        Data to fit
    x_values : array
        X-axis values
    force_direction : str or None
        'increasing' for positive slope, 'decreasing' for negative slope
    
    Returns:
    --------
    dict with keys:
        - 'L': upper asymptote
        - 'x0': inflection point
        - 'k': steepness
        - 'b': baseline
        - 'r_squared': goodness of fit
        - 'fit_success': whether fitting succeeded
        - 'fitted_curve': the fitted values
    """
    from scipy.optimize import curve_fit
    import warnings
    
    try:
        # Initial parameter guesses
        L_init = np.max(row_data) - np.min(row_data)
        x0_init = x_values[len(x_values) // 2]
        b_init = np.min(row_data)
        
        # Set k_init and bounds based on direction
        if force_direction == 'increasing':
            k_init = 0.01
            k_bounds = (0.0001, np.inf)  # Force positive k
        elif force_direction == 'decreasing':
            k_init = -0.01
            k_bounds = (-np.inf, -0.0001)  # Force negative k
        else:
            k_init = 0.01
            k_bounds = (-np.inf, np.inf)  # Allow both
        
        # Suppress overflow and invalid value warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Fit the sigmoid
            popt, pcov = curve_fit(
                sigmoid, 
                x_values, 
                row_data,
                p0=[L_init, x0_init, k_init, b_init],
                maxfev=10000,
                bounds=([0, x_values[0], k_bounds[0], -np.inf], 
                        [np.inf, x_values[-1], k_bounds[1], np.inf])
            )
        
        # Calculate R-squared
        fitted_curve = sigmoid(x_values, *popt)
        ss_res = np.sum((row_data - fitted_curve) ** 2)
        ss_tot = np.sum((row_data - np.mean(row_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'L': popt[0],           # Upper asymptote
            'x0': popt[1],          # Inflection point (midpoint)
            'k': popt[2],           # Steepness
            'b': popt[3],           # Baseline
            'r_squared': r_squared,
            'fit_success': True,
            'fitted_curve': fitted_curve,
            'peak_slope': abs(popt[0] * popt[2] / 4),  # Absolute value of max slope
            'direction': 'increasing' if popt[2] > 0 else 'decreasing'
        }
    except Exception as e:
        return {
            'L': np.nan,
            'x0': np.nan,
            'k': np.nan,
            'b': np.nan,
            'r_squared': np.nan,
            'fit_success': False,
            'fitted_curve': np.full_like(x_values, np.nan),
            'peak_slope': np.nan,
            'direction': 'failed',
            'error': str(e)
        }


def fit_sigmoids_to_df(df, lag_start=-5000, lag_end=5000):
    """
    Fit sigmoid curves to sentence and sentence2 rows in a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with label3 column and lag columns (can be named with lag values)
    lag_start : int
        Start lag value in milliseconds (default: -5000)
    lag_end : int
        End lag value in milliseconds (default: 5000)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with sigmoid fit parameters
    """
    # Get lag columns within the specified range
    lag_cols = []
    lag_vals = []
    for col in df.columns:
        if isinstance(col, (int, float, np.integer, np.floating)):
            col_val = float(col)
            if lag_start <= col_val <= lag_end:
                lag_cols.append(col)
                lag_vals.append(col_val)
    
    if len(lag_cols) == 0:
        print(f"Warning: No lag columns found in range [{lag_start}, {lag_end}]")
        return pd.DataFrame()
    
    # Sort by lag values
    sorted_indices = np.argsort(lag_vals)
    lag_cols_sorted = [lag_cols[i] for i in sorted_indices]
    lag_vals_sorted = np.array([lag_vals[i] for i in sorted_indices])
    
    # Create x_values for fitting (normalized indices)
    x_values = np.arange(len(lag_cols_sorted))
    
    sigmoid_params = []
    
    inp_df = df[df['label3'].isin(['sentence', 'sentence2'])].copy()
    counter = 0
    for idx, row in inp_df.iterrows():
        counter += 1
        print(f"Fitting row {counter}/{len(inp_df)}", end='\r')
        row_data = row[lag_cols_sorted].values
        
        # Determine direction based on label3
        if inp_df.loc[idx, 'label3'] == 'sentence':
            direction = 'increasing'
        elif inp_df.loc[idx, 'label3'] == 'sentence2':
            direction = 'decreasing'
        else:
            direction = None
        
        params = fit_sigmoid_to_row(row_data, x_values, force_direction=direction)
        params['electrode'] = inp_df.loc[idx, 'electrode']
        params['roi'] = inp_df.loc[idx, 'roi']
        params['label3'] = inp_df.loc[idx, 'label3']
        params['subject'] = inp_df.loc[idx, 'subject']
        # Store the actual lag range used
        params['lag_start'] = lag_start
        params['lag_end'] = lag_end
        sigmoid_params.append(params)
    
    print()  # New line after progress
    return pd.DataFrame(sigmoid_params)


def get_max_joint_performance(df, thresh=0.1):
    """
    Get maximum joint performance for each electrode.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with label3 column and lag columns
    thresh : float
        Threshold for joint performance
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with electrode, subject, roi, max_joint columns
    """
    joint_rows = df[df['label3'] == 'joint'].copy()
    
    # Get numeric lag columns
    lag_cols = [col for col in joint_rows.columns if isinstance(col, (int, float, np.integer, np.floating))]
    
    # Calculate max for each electrode
    joint_rows['max_joint'] = joint_rows[lag_cols].max(axis=1)
    
    # Filter by threshold
    joint_rows = joint_rows[joint_rows['max_joint'] > thresh]
    
    return joint_rows[['subject', 'electrode', 'roi', 'max_joint']]


def calculate_max_diff(df_original, df_control, label3_val, lag_cutoff, direction='future'):
    """
    Calculate difference in maximum values between original and control conditions.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original encoding results
    df_control : pd.DataFrame
        Control condition encoding results (e.g., rand_m_diff_w, opp_m_diff_w)
    label3_val : str
        'sentence' or 'sentence2'
    lag_cutoff : int
        Lag value to split future/past (typically 0)
    direction : str
        'future' (lags <= cutoff) or 'past' (lags >= cutoff)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with electrode, subject, roi, max_diff columns
    """
    # Filter by label3
    df_orig_filt = df_original[df_original['label3'] == label3_val].copy()
    df_ctrl_filt = df_control[df_control['label3'] == label3_val].copy()
    
    # Get numeric lag columns
    lag_cols = [col for col in df_orig_filt.columns if isinstance(col, (int, float, np.integer, np.floating))]
    
    # Filter lag columns by direction
    if direction == 'future':
        lag_cols_filt = [col for col in lag_cols if float(col) <= lag_cutoff]
    else:  # past
        lag_cols_filt = [col for col in lag_cols if float(col) >= lag_cutoff]
    
    results = []
    
    for idx, row_orig in df_orig_filt.iterrows():
        # Find matching electrode in control
        row_ctrl = df_ctrl_filt[
            (df_ctrl_filt['subject'] == row_orig['subject']) & 
            (df_ctrl_filt['electrode'] == row_orig['electrode'])
        ]
        
        if len(row_ctrl) == 0:
            continue
        
        row_ctrl = row_ctrl.iloc[0]
        
        # Get max values
        max_orig = row_orig[lag_cols_filt].max()
        max_ctrl = row_ctrl[lag_cols_filt].max()
        
        results.append({
            'subject': row_orig['subject'],
            'electrode': row_orig['electrode'],
            'roi': row_orig.get('roi', 'unknown'),
            'label3': label3_val,
            'direction': direction,
            'max_original': max_orig,
            'max_control': max_ctrl,
            'max_diff': max_orig - max_ctrl
        })
    
    return pd.DataFrame(results)

# Make interactive videos (mp4/html)

# Extract future and past data for specified lag range
def prepare_video_data(df, thresh_df, label3_val, lags, df_coords):
    """Prepare data for video: filter by label3, threshold, and extract specified lags."""
    # Filter by label3
    data = df[df['label3'] == label3_val].copy()
    
    # Merge with threshold info
    data = data.merge(thresh_df, on=['subject', 'electrode'], how='inner')
    
    # Merge with coordinates
    data['subject'] = data['subject'].astype(str)
    data = data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                      on=['subject', 'electrode'], 
                      how='left')
    
    # Check for missing coordinates
    missing_coords = data[['x', 'y', 'z']].isna().any(axis=1).sum()
    if missing_coords > 0:
        print(f"  Warning: {missing_coords} electrodes missing coordinates")
        data = data.dropna(subset=['x', 'y', 'z'])
    
    # Select metadata and lag columns
    meta_cols = ['subject', 'electrode', 'roi', 'x', 'y', 'z', 'max_joint']
    
    # Ensure all requested lags exist in the data
    available_lags = [lag for lag in lags if lag in data.columns]
    
    result = data[meta_cols + available_lags].copy()
    
    print(f"  {label3_val}: {len(result)} electrodes, {len(available_lags)} time points")
    
    return result

def prepare_ratio_video_data(df, thresh_df, label3_numerator, label3_denominator, lags, df_coords, joint_max=False):
    """Prepare ratio data for video: (numerator/denominator) for each lag."""
    # Filter by label3 for numerator (future or past)
    numerator_data = df[df['label3'] == label3_numerator].copy()
    
    # Merge with threshold info
    numerator_data = numerator_data.merge(thresh_df, on=['subject', 'electrode'], how='inner')
    
    # Filter by label3 for denominator (joint)
    denominator_data = df[df['label3'] == label3_denominator].copy()
    
    # Merge with threshold info for denominator
    denominator_data = denominator_data.merge(thresh_df, on=['subject', 'electrode'], how='inner')
    
    # Convert subject to string for BOTH dataframes
    numerator_data['subject'] = numerator_data['subject'].astype(str)
    denominator_data['subject'] = denominator_data['subject'].astype(str)
    
    # Merge with coordinates
    numerator_data = numerator_data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                                          on=['subject', 'electrode'], 
                                          how='left')
    
    # Check for missing coordinates
    missing_coords = numerator_data[['x', 'y', 'z']].isna().any(axis=1).sum()
    if missing_coords > 0:
        # print(f"  Warning: {missing_coords} electrodes missing coordinates")
        numerator_data = numerator_data.dropna(subset=['x', 'y', 'z'])
    
    # Select metadata columns
    meta_cols = ['subject', 'electrode', 'roi', 'x', 'y', 'z', 'max_joint']
    
    # Ensure all requested lags exist in the data
    available_lags = [lag for lag in lags if lag in numerator_data.columns]
    
    # Create result dataframe with metadata
    result = numerator_data[meta_cols].copy()
    
    if joint_max:
        # Calculate the maximum joint value across all lags for each electrode
        denominator_data['joint_max'] = denominator_data[available_lags].max(axis=1)
        denominator_aligned = denominator_data[['subject', 'electrode', 'joint_max']].copy()
    else:
        # Use the joint value at each lag
        denominator_aligned = denominator_data[['subject', 'electrode'] + available_lags].copy()
    
    # Rename lag columns in numerator and denominator for clarity
    numerator_aligned = numerator_data[['subject', 'electrode'] + available_lags].copy()
    numerator_aligned = numerator_aligned.rename(columns={lag: f"{lag}_num" for lag in available_lags})
    
    if not joint_max:
        denominator_aligned = denominator_aligned.rename(columns={lag: f"{lag}_denom" for lag in available_lags})
    
    # Merge to ensure alignment
    merged = numerator_aligned.merge(
        denominator_aligned, 
        on=['subject', 'electrode'], 
        how='inner'
    )
    
    # Calculate ratio for each lag
    for lag in available_lags:
        if joint_max:
            denom_col = 'joint_max'
        else:
            denom_col = f'{lag}_denom'
        
        num_col = f'{lag}_num'
        
        # Calculate ratio (handle division by zero)
        ratio = merged[num_col] / merged[denom_col].replace(0, np.nan)
        
        # Merge ratio back into result using subject-electrode match
        ratio_df = merged[['subject', 'electrode']].copy()
        ratio_df[lag] = ratio.values
        
        result = result.merge(ratio_df, on=['subject', 'electrode'], how='left')
    
    print(f"  {label3_numerator}/{label3_denominator}: {len(result)} electrodes, {len(available_lags)} time points")
    
    return result

def prepare_all_ratio_video_data(df, thresh_df, lags, df_coords, joint_max=True):
    """Calculate future, word, and past ratios simultaneously to ensure consistency."""
    
    print("Preparing all ratio video data...")
    
    # Get the three numerators and the denominator
    future_data = df[df['label3'] == 'sentence'].copy()
    word_data = df[df['label3'] == 'word'].copy()
    past_data = df[df['label3'] == 'sentence2'].copy()
    joint_data = df[df['label3'] == 'joint'].copy()
    
    # Convert subject to string for ALL dataframes (including thresh_df and df_coords)
    thresh_df = thresh_df.copy()
    thresh_df['subject'] = thresh_df['subject'].astype(str)
    
    df_coords = df_coords.copy()
    df_coords['subject'] = df_coords['subject'].astype(str)
    
    # Convert subject to string in data
    for data in [future_data, word_data, past_data, joint_data]:
        data['subject'] = data['subject'].astype(str)
    
    # Merge with thresholds and coordinates (reassign, not inplace)
    future_data = future_data.merge(thresh_df[['subject', 'electrode', 'max_joint']], 
                                     on=['subject', 'electrode'], how='inner')
    future_data = future_data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                                     on=['subject', 'electrode'], how='left')
    
    word_data = word_data.merge(thresh_df[['subject', 'electrode', 'max_joint']], 
                                 on=['subject', 'electrode'], how='inner')
    word_data = word_data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                                 on=['subject', 'electrode'], how='left')
    
    past_data = past_data.merge(thresh_df[['subject', 'electrode', 'max_joint']], 
                                 on=['subject', 'electrode'], how='inner')
    past_data = past_data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                                 on=['subject', 'electrode'], how='left')
    
    joint_data = joint_data.merge(thresh_df[['subject', 'electrode', 'max_joint']], 
                                   on=['subject', 'electrode'], how='inner')
    joint_data = joint_data.merge(df_coords[['subject', 'electrode', 'x', 'y', 'z']], 
                                   on=['subject', 'electrode'], how='left')
    
    # Drop rows with missing coordinates
    future_data = future_data.dropna(subset=['x', 'y', 'z'])
    word_data = word_data.dropna(subset=['x', 'y', 'z'])
    past_data = past_data.dropna(subset=['x', 'y', 'z'])
    joint_data = joint_data.dropna(subset=['x', 'y', 'z'])
    
    # Available lags
    available_lags = [lag for lag in lags if lag in joint_data.columns]
    
    # Calculate joint denominator
    if joint_max:
        # Use maximum joint value across all lags for each electrode
        joint_data['joint_denom'] = joint_data[available_lags].max(axis=1)
    
    # Create base dataframe with metadata (using joint_data as reference)
    base = joint_data[['subject', 'electrode', 'roi', 'x', 'y', 'z', 'max_joint']].copy()
    
    # Set index for efficient lookup
    future_indexed = future_data.set_index(['subject', 'electrode'])
    word_indexed = word_data.set_index(['subject', 'electrode'])
    past_indexed = past_data.set_index(['subject', 'electrode'])
    joint_indexed = joint_data.set_index(['subject', 'electrode'])
    
    results = {}
    for name, data_indexed in [('future', future_indexed), ('word', word_indexed), ('past', past_indexed)]:
        result = base.copy()
        
        for lag in available_lags:
            # Get numerator values
            num = data_indexed[lag].reindex(result.set_index(['subject', 'electrode']).index)
            
            # Get denominator values
            if joint_max:
                denom = joint_indexed['joint_denom'].reindex(result.set_index(['subject', 'electrode']).index)
            else:
                denom = joint_indexed[lag].reindex(result.set_index(['subject', 'electrode']).index)
            
            # Calculate ratio
            result[lag] = (num / denom.replace(0, np.nan)).values
        
        results[name] = result
        print(f"  {name}: {len(result)} electrodes, {len(available_lags)} lags")
    
    return results['future'], results['word'], results['past']

def create_interactive_brain_viz_html(future_df, word_df, past_df, lags, output_path, title_prefix="Comprehension", vmin=0.00, vmax=0.25):
    """
    Create an interactive HTML visualization using pre-rendered nilearn glass brain images.
    Each frame is rendered as a matplotlib figure and embedded in Plotly.
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import io
    import base64
    from PIL import Image
    from nilearn import plotting

    print(f"Creating interactive HTML visualization with glass brains: {title_prefix}")
    print(f"  Rendering {len(lags)} frames...")
    
    # Pre-render all frames as images
    image_data = []
    
    for lag_idx, lag in enumerate(lags):
        if lag_idx % 5 == 0:
            print(f"  Rendering frame {lag_idx+1}/{len(lags)}")
        
        # Create figure with 3 rows
        fig_mpl, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Future (row 1)
        plotting.plot_markers(
            node_values=future_df[lag].values,
            node_coords=future_df[['x', 'y', 'z']].values,
            node_size=50,
            node_cmap='Greens',
            node_vmin=vmin,
            node_vmax=vmax,
            display_mode='lzry',
            colorbar=True,
            axes=axes[0],
            title=f"Future (Sentence) - Lag: {lag} ms"
        )
        
        # Word (row 2)
        plotting.plot_markers(
            node_values=word_df[lag].values,
            node_coords=word_df[['x', 'y', 'z']].values,
            node_size=50,
            node_cmap='Oranges',
            node_vmin=vmin,
            node_vmax=vmax,
            display_mode='lzry',
            colorbar=True,
            axes=axes[1],
            title=f"Word - Lag: {lag} ms"
        )
        
        # Past (row 3)
        plotting.plot_markers(
            node_values=past_df[lag].values,
            node_coords=past_df[['x', 'y', 'z']].values,
            node_size=50,
            node_cmap='Reds',
            node_vmin=vmin,
            node_vmax=vmax,
            display_mode='lzry',
            colorbar=True,
            axes=axes[2],
            title=f"Past (Sentence2) - Lag: {lag} ms"
        )
        
        plt.suptitle(f"{title_prefix} - Lag: {lag} ms", fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Convert to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_mpl)
        
        image_data.append(f"data:image/png;base64,{img_base64}")
    
    print("  Creating interactive Plotly figure...")
    
    # Create Plotly figure with image frames
    fig = go.Figure()
    
    # Add initial frame
    fig.add_layout_image(
        dict(
            source=image_data[0],
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            layer="below"
        )
    )
    
    # Create frames
    frames = [
        go.Frame(
            layout=go.Layout(
                images=[dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=0,
                    y=1,
                    sizex=1,
                    sizey=1,
                    sizing="stretch",
                    layer="below"
                )],
                title=f"{title_prefix} - Lag: {lag} ms"
            ),
            name=str(lag)
        )
        for img, lag in zip(image_data, lags)
    ]
    
    fig.frames = frames
    
    # Add slider
    sliders = [dict(
        active=0,
        yanchor="top",
        y=-0.1,
        xanchor="left",
        currentvalue=dict(
            prefix="Lag (ms): ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=300),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        steps=[dict(
            args=[[f.name], dict(
                frame=dict(duration=300, redraw=True),
                mode="immediate",
                transition=dict(duration=300)
            )],
            label=str(lag),
            method="animate"
        ) for f, lag in zip(frames, lags)]
    )]
    
    # Add play/pause buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        x=0.05,
        y=-0.15,
        xanchor="left",
        yanchor="top",
        buttons=[
            dict(label="▶ Play",
                 method="animate",
                 args=[None, dict(
                     frame=dict(duration=500, redraw=True),
                     fromcurrent=True,
                     transition=dict(duration=300)
                 )]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode="immediate",
                     transition=dict(duration=0)
                 )])
        ]
    )]
    
    # Update layout
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    
    fig.update_layout(
        title=f"{title_prefix} - Brain Activity Over Time",
        height=900,
        width=1000,
        sliders=sliders,
        updatemenus=updatemenus,
        margin=dict(l=0, r=0, t=50, b=150),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"  ✓ Saved interactive HTML to: {output_path}\n")
    
    return fig

def plot_glassbrain_frame(df, lag, cmap, title, vmin=-0.1, vmax=0.3):
    """Plot a single glass brain frame for a specific lag."""
    # Get coordinates and values for this lag
    coords = df[['x', 'y', 'z']].values
    values = df[lag].values
    
    # Create figure
    fig = plt.figure(figsize=(12, 4))
    
    # Plot glass brain
    display = plotting.plot_markers(
        node_values=values,
        node_coords=coords,
        node_size=50,
        node_cmap=cmap,
        node_vmin=vmin,
        node_vmax=vmax,
        display_mode='lzry',
        colorbar=True,
        figure=fig
    )
    
    # Add title with lag information
    plt.suptitle(f"{title}\nLag: {lag} ms", fontsize=14, y=0.98)
    
    return fig

def create_video(df, lags, cmap, title, output_path, vmin=-0.1, vmax=0.3, fps=10):
    """Create video from glass brain frames across lags."""
    import matplotlib.animation as animation
    from nilearn import plotting

    print(f"Creating video: {title}")
    print(f"  Frames: {len(lags)}")
    print(f"  Output: {output_path}")
    
    # Get coordinates (same for all frames)
    coords = df[['x', 'y', 'z']].values
    
    # Set up the figure and animation
    fig = plt.figure(figsize=(12, 4))
    
    def update_frame(frame_idx):
        """Update function for animation."""
        lag = lags[frame_idx]
        values = df[lag].values
        
        # Clear previous frame
        plt.clf()
        
        # Plot new frame
        display = plotting.plot_markers(
            node_values=values,
            node_coords=coords,
            node_size=50,
            node_cmap=cmap,
            node_vmin=vmin,
            node_vmax=vmax,
            display_mode='lzry',
            colorbar=True,
            figure=fig
        )
        
        plt.suptitle(f"{title}\nLag: {lag} ms", fontsize=14, y=0.98)
        
        return fig,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=len(lags),
        interval=1000/fps,  # milliseconds per frame
        blit=False
    )
    
    # Save video
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"  Video saved!\n")
    
    return anim


def calc_max_diff_percentile(dfs, percentile=99.5):
    """
    Calculate the maximum absolute difference value at a given percentile across multiple dataframes.
    
    Parameters:
    -----------
    dfs : list of DataFrames
        List of dataframes to calculate percentiles from
    percentile : float
        Percentile to use (default 99.5)
        
    Returns:
    --------
    float
        Maximum absolute value at the specified percentile
    """
    max_vals = []
    for df in dfs:
        # Get all numeric columns (lag columns)
        numeric_cols = [col for col in df.columns if col not in ['subject', 'electrode', 'roi', 'label3', 'x', 'y', 'z']]
        if numeric_cols:
            vals = df[numeric_cols].abs().values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                max_vals.append(np.percentile(vals, percentile))
    
    return max(max_vals) if max_vals else 1.0


def calc_max_diff_percentile_with_baseline(
    df_control,
    df_reph,
    label3_val,
    time_window_cols,
    percentile=90,
    min_diff=None,
    min_baseline_score=None,
):
    """
    Compute per-electrode max |control - reph| within a lag window, 
    with additional filtering based on baseline encoding scores.
    
    This function filters electrodes based on two criteria:
    1. The difference between control and rephrased conditions (same as calc_max_diff_percentile)
    2. The highest encoding score in the time window for EITHER condition must be above a threshold
    
    Parameters:
    -----------
    df_control : DataFrame
        Control condition dataframe
    df_reph : DataFrame
        Rephrased condition dataframe
    label3_val : str
        Label to filter on (e.g., 'sentence', 'sentence2')
    time_window_cols : list
        List of lag columns to consider in the time window
    percentile : float
        Percentile to use for thresholding (default 90)
    min_diff : float, optional
        Minimum absolute difference threshold. Electrodes with max_diff < min_diff are excluded.
    min_baseline_score : float, optional
        Minimum baseline encoding score threshold. Electrodes where the maximum score
        in the time window for BOTH conditions is below this threshold are excluded.
        If None, no baseline filtering is applied.
        
    Returns:
    --------
    DataFrame
        Results dataframe with columns: subject, subject_electrode, electrode, label3,
        max_diff, max_diff_lag, roi, percentile, percentile_threshold, above_percentile,
        max_control_score, max_reph_score, baseline_filter_passed
    """
    results = []

    df_control_filt = df_control[df_control["label3"] == label3_val].copy()
    df_reph_filt    = df_reph[df_reph["label3"] == label3_val].copy()

    # Prefer a stable unique key if present (these result files use subject_electrode)
    if "subject_electrode" in df_control_filt.columns and "subject_electrode" in df_reph_filt.columns:
        merge_keys = ["subject_electrode"]
    elif (
        "subject" in df_control_filt.columns and "subject" in df_reph_filt.columns
        and "electrode" in df_control_filt.columns and "electrode" in df_reph_filt.columns
    ):
        merge_keys = ["subject", "electrode"]
    elif "electrode" in df_control_filt.columns and "electrode" in df_reph_filt.columns:
        merge_keys = ["electrode"]
    else:
        raise KeyError("No common merge key found (expected subject_electrode or subject+electrode or electrode)")

    merged = df_control_filt.merge(
        df_reph_filt,
        on=merge_keys,
        suffixes=("_control", "_reph"),
    )

    if min_diff is not None:
        try:
            min_diff = float(min_diff)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_diff must be a number or None, got: {min_diff!r}") from exc
    
    if min_baseline_score is not None:
        try:
            min_baseline_score = float(min_baseline_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_baseline_score must be a number or None, got: {min_baseline_score!r}") from exc

    for _, row in merged.iterrows():
        control_cols = [f"{col}_control" for col in time_window_cols if f"{col}_control" in row.index]
        reph_cols    = [f"{col}_reph" for col in time_window_cols if f"{col}_reph" in row.index]

        control_vals = row[control_cols].to_numpy()
        reph_vals    = row[reph_cols].to_numpy()

        diffs = np.abs(control_vals - reph_vals)

        max_idx = int(np.argmax(diffs)) if len(diffs) else 0
        max_diff = float(np.max(diffs)) if len(diffs) else np.nan
        max_diff_lag = time_window_cols[max_idx] if len(time_window_cols) else np.nan

        # Calculate max scores for each condition in the time window
        max_control_score = float(np.max(control_vals)) if len(control_vals) else np.nan
        max_reph_score = float(np.max(reph_vals)) if len(reph_vals) else np.nan

        # Check baseline score threshold (either condition must pass)
        baseline_filter_passed = True
        if min_baseline_score is not None:
            # At least one condition must have a score above the threshold
            if (not np.isfinite(max_control_score) or max_control_score < min_baseline_score) and \
               (not np.isfinite(max_reph_score) or max_reph_score < min_baseline_score):
                baseline_filter_passed = False

        # Apply difference threshold filter
        if min_diff is not None:
            if (not np.isfinite(max_diff)) or (max_diff < min_diff):
                continue

        # Apply baseline filter
        if not baseline_filter_passed:
            continue

        subject_electrode = row.get("subject_electrode", np.nan)
        electrode = row.get("electrode_control", row.get("electrode_reph", row.get("electrode", np.nan)))
        roi = row.get("roi_control", row.get("roi_reph", "unknown"))

        # plot_effect_glassbrain expects a 'subject' column
        if pd.notna(subject_electrode):
            subject = str(subject_electrode).split("_", 1)[0]
        else:
            subject = row.get("subject", row.get("subject_control", row.get("subject_reph", np.nan)))

        results.append({
            "subject": subject,
            "subject_electrode": subject_electrode,
            "electrode": electrode,
            "label3": label3_val,
            "max_diff": max_diff,
            "max_diff_lag": max_diff_lag,
            "roi": roi,
            "max_control_score": max_control_score,
            "max_reph_score": max_reph_score,
            "baseline_filter_passed": baseline_filter_passed,
        })

    results_df = pd.DataFrame(results)

    # Ensure these exist for all percentiles (concat-friendly)
    results_df["percentile"] = int(percentile)
    results_df["percentile_threshold"] = np.nan
    results_df["above_percentile"] = False

    if len(results_df) > 0 and results_df["max_diff"].notna().any():
        thr = np.percentile(results_df["max_diff"].dropna().to_numpy(), percentile)
        results_df["percentile_threshold"] = thr
        results_df["above_percentile"] = results_df["max_diff"] >= thr

    return results_df


def plot_all_lines_across_datasets(args, sig_results_dict, roi, mode="comp", ymax=0.18, save=False, titles=None):
    """
    Plot all lines (joint, word, sentence, sentence2) across multiple datasets in a row of subplots.
    Each subplot shows one line type with different datasets in different shades.
    
    Parameters:
    -----------
    args : Args object
        Contains configuration parameters
    sig_results_dict : dict
        Dictionary keyed by (i, line, roi) containing DataFrames
    roi : str
        ROI name to plot
    mode : str
        "comp" or "prod"
    ymax : float
        Maximum y-axis value
    save : bool
        Whether to save the figure
    titles : list of str, optional
        Custom titles for each dataset. If None, uses generic titles.
    """
    # Determine number of datasets from keys
    dataset_indices = sorted(set(key[0] for key in sig_results_dict.keys()))
    n_datasets = len(dataset_indices)
    
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(n_datasets)]
    
    # Create figure with 4 subplots in a row
    n_lines = len(args.lines)
    fig, axes = plt.subplots(1, n_lines, figsize=(20, 5))
    
    # Plot each line type in its own subplot
    for line_idx, line in enumerate(args.lines):
        ax = axes[line_idx]
        
        # Get base color for this line
        base_color = args.colors[line_idx]
        
        # Darken base color if more than 3 datasets for better contrast
        if n_datasets > 3:
            import matplotlib.colors as mcolors
            # Convert to RGB, darken by 20%, then back to hex/name
            rgb = mcolors.to_rgb(base_color)
            darkened_rgb = tuple(max(0, c * 0.8) for c in rgb)
            base_color = darkened_rgb
        
        # Plot each dataset with different opacity/shade (REVERSED)
        for i, dataset_idx in enumerate(dataset_indices):
            # Create progressively darker/lighter shades with higher contrast (REVERSED ORDER)
            alpha = 1.0 - (0.8 * i / max(1, n_datasets - 1))  # Range from 1.0 to 0.2 (increased contrast)
            
            key = (dataset_idx, line, roi)
            
            if key not in sig_results_dict or sig_results_dict[key].empty:
                continue
            
            df_line = sig_results_dict[key]
            
            # Count electrodes
            n_elecs = len(df_line)
            label = f"{titles[i]} (n={n_elecs})"
            
            # Plot line with varying alpha
            lags_selected = [lag_idx for lag_idx, lag in enumerate(args.lags["lags_all"]) 
                            if lag in args.lags["lags_plt"]]
            vals = df_line.iloc[:, lags_selected].mean(axis=0)
            errs = df_line.iloc[:, lags_selected].sem(axis=0)
            
            ax.plot(
                args.lags["lags_plt"],
                vals,
                color=base_color,
                alpha=alpha,
                label=label,
                lw=2.0, 
            )
            ax.fill_between(
                args.lags["lags_plt"],
                vals - errs,
                vals + errs,
                alpha=alpha * 0.25, 
                color=base_color,
            )
        
        # Ax cosmetics
        if ymax is not None:
            ax.set_ylim(top=ymax, bottom=-0.025)
        ymin, ymax_val = ax.get_ylim()
        
        # Shade time window
        if mode == "comp":
            rect1 = patches.Rectangle((50, ymin), 450, ymax_val - ymin, 
                                      color="yellowgreen", alpha=0.3, label="_nolegend_")
        elif mode == "prod":
            rect1 = patches.Rectangle((-500, ymin), 450, ymax_val - ymin, 
                                      color="indianred", alpha=0.3, label="_nolegend_")
        ax.add_patch(rect1)
        
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_xticks(args.lags["lag_ticks"])
        ax.set_xticklabels(args.lags["lag_tick_labels"])
        ax.tick_params(axis='both', which='both', labelsize=10)
        
        ax.legend(loc="best", frameon=False, fontsize=9)
        
        mode_full = "Comprehension" if mode == "comp" else "Production"
        ax.set_title(f"{args.legends[line_idx]}", fontsize=12)
        
    
    # Add overall title
    mode_full = "Comprehension" if mode == "comp" else "Production"
    fig.suptitle(f"{mode_full} - {roi}", fontsize=14, y=1.02)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{args.res_dir}/{roi}_all_lines_overlay_{mode}.jpeg", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
