import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import tfsplt_future_past_utils as pu


class Args(argparse.Namespace):
    rois = [
        "IFG",
        "MTG",
        "STG",
        "preCG",
        "postCG",
        "rostralmiddlefrontal",
        "superiorfrontal",
        "supramarginal",
        "ITG",
        "TP",
        "fusiform",
        "All",
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legends = ["joint"]
    lines = ["joint"]
    lags_full = np.arange(-30000, 30001, 50)
    lags = {
        "lags_all": lags_full,
        "lags_plt": lags_full,
        "lag_ticks": np.arange(-30000, 30001, 5000),
        "lag_tick_labels": np.arange(-30, 30.001, 5),
    }
    res_dir = (
        "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/roi_figures/"
        "future_past/context_joint_only/plots"
    )


def _lag_columns_for_plot(df, lags_obj):
    chosen = []
    for lag_val in lags_obj["lags_plt"]:
        # index-based (0..N-1)
        if lag_val in lags_obj["lags_all"]:
            idx = np.where(np.array(lags_obj["lags_all"]) == lag_val)[0][0]
            idx_str = str(idx)
            if idx_str in df.columns:
                chosen.append(idx_str)
                continue
            if idx in df.columns:
                chosen.append(idx)
                continue
        # raw lag value as string or numeric
        lag_str = str(lag_val)
        lag_str_float = f"{float(lag_val):.1f}"
        if lag_str in df.columns:
            chosen.append(lag_str)
            continue
        if lag_str_float in df.columns:
            chosen.append(lag_str_float)
            continue
        if lag_val in df.columns:
            chosen.append(lag_val)
            continue
    return chosen


def trim_lags_to_window(df, start_idx, end_idx):
    """Keep only lag columns in the numeric range [start_idx, end_idx], plus metadata."""
    keep_lag_cols = [str(i) for i in range(start_idx, end_idx + 1) if str(i) in df.columns]
    meta_cols = [c for c in df.columns if not c.isdigit()]
    df_trim = df[keep_lag_cols + meta_cols].copy()
    rename_map = {old: str(idx) for idx, old in enumerate(keep_lag_cols)}
    df_trim = df_trim.rename(columns=rename_map)
    return df_trim


def filter_by_shared_electrodes(dfs, shared_electrodes):
    return [df[df["electrode"].astype(str).isin(shared_electrodes)].copy() for df in dfs]


def _electrodes_over_thresh(df, label, thresh):
    df_label = df[df["label3"] == label]
    if df_label.empty:
        return set()
    lag_cols = [c for c in df_label.columns if str(c).isdigit()]
    if not lag_cols:
        return set()
    mask = df_label[lag_cols].max(axis=1) > thresh
    return set(df_label.loc[mask, "electrode"].astype(str))


def shared_elecs_or_context_and_word(context_dfs, word_dfs, thresh=0.1):
    elecs = set()
    for df in context_dfs:
        elecs |= _electrodes_over_thresh(df, "joint", thresh)
    for df in word_dfs:
        elecs |= _electrodes_over_thresh(df, "word", thresh)
    return elecs


def plot_roi_all_models(
    datasets,
    lags_obj,
    roi,
    mode,
    outfile,
    line_colors,
    ymax=0.18,
    pvals_df=None,
    sig_fdr=False,
    sig_alpha=0.05,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    n_datasets = len(datasets)
    linestyles = ["-", "--", "-.", ":"]

    for i, (name, df, line_label) in enumerate(datasets):
        if df is None or df.empty:
            continue
        cols = _lag_columns_for_plot(df, lags_obj)
        if not cols:
            continue
        vals = df[cols].mean(axis=0)
        errs = df[cols].sem(axis=0)
        base_color = line_colors.get(name, line_colors.get(line_label, "black"))
        alpha = 1.0 - 0.7 * (i / max(1, n_datasets - 1))
        ax.plot(
            lags_obj["lags_plt"],
            vals,
            color=base_color,
            alpha=alpha,
            lw=1.2,
            linestyle=linestyles[i % len(linestyles)],
            label=name,
        )
        ax.fill_between(lags_obj["lags_plt"], vals - errs, vals + errs, color=base_color, alpha=0.08 * alpha)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ymin, ymax_val = ax.get_ylim()
    sig_top = 0.25
    if pvals_df is not None and not pvals_df.empty:
        sig_top = 0.24
    if ymax is not None:
        ax.set_ylim(top=max(ymax, sig_top), bottom=min(ymin, -0.025))
        ymin, ymax_val = ax.get_ylim()

    if mode == "comp":
        rect = patches.Rectangle((50, ymin), 450, ymax_val - ymin, color="yellowgreen", alpha=0.25, label="_nolegend_")
    else:
        rect = patches.Rectangle((-500, ymin), 450, ymax_val - ymin, color="indianred", alpha=0.25, label="_nolegend_")
    ax.add_patch(rect)

    if pvals_df is not None and not pvals_df.empty:
        for i, (name, df, line_label) in enumerate(datasets):
            if df is None or df.empty:
                continue

            selected_subject_electrodes = {
                f"{str(r['subject'])}_{str(r['electrode'])}"
                for _, r in df[["subject", "electrode"]].drop_duplicates().iterrows()
            }
            if not selected_subject_electrodes:
                continue

            sig_df = stouffers_roi_test(
                pvals_df=pvals_df,
                mode=mode,
                roi=roi,
                lags_obj=lags_obj,
                selected_subject_electrodes=selected_subject_electrodes,
                fdr_correct=sig_fdr,
                alpha=sig_alpha,
            )
            if sig_df.empty:
                continue

            sig_lags = sig_df.loc[sig_df["reject"], "lag_ms"].astype(int).values
            if len(sig_lags) == 0:
                continue

            lag_arr = np.array(lags_obj["lags_plt"], dtype=float)
            sig_x = [lag for lag in sig_lags if lag_arr.min() <= lag <= lag_arr.max()]
            if not sig_x:
                continue

            y_marker = 0.24 - 0.01 * i
            y_vals = np.full(len(sig_x), y_marker)
            base_color = line_colors.get(name, line_colors.get(line_label, "black"))
            ax.scatter(sig_x, y_vals, color=base_color, s=3, marker="o", zorder=6)

    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.set_xticks(lags_obj["lag_ticks"])
    ax.set_xticklabels(lags_obj["lag_tick_labels"])
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Correlation")
    ax.legend(loc="best", frameon=True, fontsize=10)
    mode_full = "Comprehension" if mode == "comp" else "Production"
    ax.set_title(f"{roi} {mode_full} (all models)", fontsize=10)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def export_electrodes_pdf_all_models(datasets, lags_obj, outfile, line_colors):
    """
    Plot all models (contexts + word-only) on the same plot for each electrode.

    datasets: list of (name, df, line_label)
    line_colors: dict mapping line_label -> base color
    """
    if not datasets:
        print(f"No datasets provided for {outfile}")
        return

    elec_keys = set()
    for _, df, _ in datasets:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            elec_keys.add((str(row.get("subject", "")), str(row.get("electrode", "")), str(row.get("roi", ""))))

    if not elec_keys:
        print(f"No electrodes to save for {outfile}")
        return

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with PdfPages(outfile) as pdf:
        for subj, elec, roi in sorted(elec_keys):
            fig, ax = plt.subplots(figsize=(10, 4))
            plotted = False
            linestyles = ["-", "--", "-.", ":"]

            for i, (name, df, line_label) in enumerate(datasets):
                if df is None or df.empty:
                    continue
                row = df[(df["subject"].astype(str) == subj) & (df["electrode"].astype(str) == elec)]
                if row.empty:
                    continue

                cols = _lag_columns_for_plot(row, lags_obj)
                if not cols:
                    continue

                vals = np.asarray(row.iloc[0][cols], dtype=float)
                base_color = line_colors.get(name, line_colors.get(line_label, "black"))
                alpha = 1.0 - 0.7 * (i / max(1, len(datasets) - 1))
                ax.plot(
                    lags_obj["lags_plt"],
                    vals,
                    color=base_color,
                    alpha=alpha,
                    lw=1.0,
                    linestyle=linestyles[i % len(linestyles)],
                    label=f"{name} ({line_label})",
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.axvline(0, color="k", ls="--", alpha=0.4)
            ax.axhline(0, color="k", ls=":", alpha=0.4)
            ax.set_xlim(lags_obj["lags_plt"][0], lags_obj["lags_plt"][-1])
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("Correlation")
            ax.set_title(f"ROI {roi} | subj {subj} elec {elec}")
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            pdf.savefig(fig, dpi=400)
            plt.close(fig)

    print(f"Saved {len(elec_keys)} electrodes to {outfile}")


def calculate_fwhm(series, lags, sigma=3):
    """
    Calculate FWHM on a smoothed version of the series.
    """
    if series.isnull().all():
        return np.nan

    smoothed = gaussian_filter1d(series.values, sigma=sigma)
    idx_max = np.argmax(smoothed)
    max_val = smoothed[idx_max]

    if max_val <= 0:
        return np.nan

    half_max = max_val / 2.0

    # Left crossing (last point before peak < half max)
    left_part = smoothed[:idx_max]
    under_half_left = np.where(left_part < half_max)[0]
    if len(under_half_left) == 0:
        idx_left = 0
    else:
        idx_left = under_half_left[-1]

    # Right crossing (first point after peak < half max)
    right_part = smoothed[idx_max:]
    under_half_right = np.where(right_part < half_max)[0]
    if len(under_half_right) == 0:
        idx_right = len(smoothed) - 1
    else:
        idx_right = idx_max + under_half_right[0]

    # Linear interpolation for better precision
    def get_exact_idx(i_below, i_above, target, data):
        y1 = data[i_below]
        y2 = data[i_above]
        if y2 == y1:
            return float(i_below)
        fraction = (target - y1) / (y2 - y1)
        return i_below + fraction

    # Left interpolation
    if idx_left < idx_max:  # Normal case
        # idx_left is < HM. The next point (idx_left+1) is >= HM (or is peak)
        idx_left_exact = get_exact_idx(idx_left, idx_left + 1, half_max, smoothed)
    else:
        idx_left_exact = float(idx_left)

    # Right interpolation
    if idx_right > idx_max:
        # idx_right is < HM. The prev point (idx_right-1) is >= HM
        idx_right_exact = get_exact_idx(idx_right, idx_right - 1, half_max, smoothed)
    else:
        idx_right_exact = float(idx_right)

    # Map indices to lags
    lag_left = np.interp(idx_left_exact, np.arange(len(lags)), lags)
    lag_right = np.interp(idx_right_exact, np.arange(len(lags)), lags)

    return lag_right - lag_left


def plot_fwhm_analysis(datasets, lags_obj, rois, out_dir, mode, model_filter=None):
    """
    Compute FWHM for each electrode.
    Make swarmplots by ROI.
    Make glassbrain topography plots.
    """
    print(f"Calculating FWHM and plotting for {mode}...")

    records = []
    lags_plt = np.array(lags_obj["lags_plt"])

    for model_name, df, _ in datasets:
        if model_filter and model_name not in model_filter:
            continue
        if df is None or df.empty:
            continue

        print(f"  Processing {model_name}...")

        # Re-resolve lags for these columns to ensure alignment
        final_lags = []
        final_cols = []
        for lag_val in lags_plt:
            idx = np.where(np.array(lags_obj["lags_all"]) == lag_val)[0][0]
            val_candidates = [
                str(idx),
                idx,
                str(lag_val),
                f"{float(lag_val):.1f}",
                lag_val,
            ]
            found_col = None
            for cand in val_candidates:
                if cand in df.columns:
                    found_col = cand
                    break

            if found_col:
                final_lags.append(lag_val)
                final_cols.append(found_col)

        if not final_cols:
            continue

        final_lags = np.array(final_lags)

        # Filter by ROIs (if needed, but for brainmaps we might want all legitimate electrodes)
        # However, we only care about the passed ROIs for swarmplots. 
        # For brainmaps, arguably we want all significant elecs.
        # But 'df' assumes rows are electrodes.
        
        # Taking all rows in df regardless of ROI for calculation:
        # (df might already be some subset effectively, but let's iterate all rows)
        
        # Use all available electrodes (already thresholded) instead of filtering by ROIs
        df_subset = df.copy()

        for _, row in df_subset.iterrows():
            vals = row[final_cols].values.astype(float)
            fwhm = calculate_fwhm(pd.Series(vals), final_lags)

            # Ensure minimal validity
            if not np.isnan(fwhm) and fwhm > 0:
                records.append({
                    "Model": model_name,
                    "ROI": row["roi"],
                    "FWHM": fwhm,
                    "Max_Performance": vals.max(),
                    "subject": row["subject"],
                    "electrode": row["electrode"]
                })

    if not records:
        print(f"No valid FWHM data found for {mode}")
        return

    fwhm_df = pd.DataFrame(records)
    fwhm_df["subject"] = fwhm_df["subject"].astype(str)
    fwhm_df["electrode"] = fwhm_df["electrode"].astype(str)
    
    # Save CSV
    csv_out_path = os.path.join(out_dir, f"FWHM_and_Max_Performance_{mode}.csv")
    fwhm_df.to_csv(csv_out_path, index=False)
    print(f"Saved FWHM and max performance values to {csv_out_path}")

    # 1. Swarmplots
    swarm_dir = os.path.join(out_dir, "fwhm_swarms")
    os.makedirs(swarm_dir, exist_ok=True)
    
    # 2. Brainmaps
    brain_dir = os.path.join(out_dir, "fwhm_brainmaps")
    os.makedirs(brain_dir, exist_ok=True)

    models = fwhm_df["Model"].unique()
    for model in models:
        subset = fwhm_df[fwhm_df["Model"] == model]
        if subset.empty:
            continue

        # -- Swarmplot --
        plt.figure(figsize=(12, 6))
        roi_order = [r for r in rois if r in subset["ROI"].unique()]
        sns.swarmplot(data=subset, x="ROI", y="FWHM", order=roi_order, size=4)
        plt.title(f"FWHM Distribution - {model} ({mode})")
        plt.ylabel("FWHM (ms)")
        plt.xlabel("Region of Interest")
        plt.xticks(rotation=45)
        out_name_swarm = os.path.join(swarm_dir, f"FWHM_{mode}_{model.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(out_name_swarm)
        plt.close()
        print(f"Saved FWHM swarm: {out_name_swarm}")

        # -- Brainmap --
        try:
            import matplotlib.colors as mcolors
            import copy
            
            mean_fwhm = subset["FWHM"].mean()
            
            subjects_in_subset = subset["subject"].unique().tolist()
            # Remap 7170 -> 717 for coordinate loading
            subjects_for_loading = [s if s != "7170" else "717" for s in subjects_in_subset]
            subjects_for_loading = list(set(subjects_for_loading))
            
            node_kwargs_outline = {"edgecolors": "black", "linewidths": 0.5}
            orig_cmap = copy.copy(plt.get_cmap("PRGn"))

            # 1. Version with mean explicitly centered in the colormap but range [0, 17000]
            for cmap_name in ["PRGn", "PuOr"]:
                cmap_base = copy.copy(plt.get_cmap(cmap_name))
                norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=mean_fwhm, vmax=17000)
                shifted_colors = cmap_base(norm(np.linspace(0, 17000, 256)))
                shifted_cmap = mcolors.ListedColormap(shifted_colors, name=f"{cmap_name}_shifted")

                out_name_brain1 = os.path.join(brain_dir, f"FWHM_brain_{mode}_{model.replace(' ', '_')}_{cmap_name}_mean_center.png")
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                pu.plot_effect_glassbrain(
                    subset,
                    effect_col="FWHM",
                    subjects=subjects_for_loading,
                    cmap=shifted_cmap,
                    outfile=out_name_brain1,
                    vmin=0,
                    vmax=17000,
                    ax=ax1, 
                    show=False,
                    title=f"Continuous FWHM - {model} ({mode}) | center={mean_fwhm:.0f}ms",
                    node_kwargs=node_kwargs_outline,
                )
                fig1.savefig(out_name_brain1)
                fig1.savefig(os.path.splitext(out_name_brain1)[0] + ".svg")
                plt.close(fig1)

            # 2. Version with 5 bins covering 0:17000
            discrete_colors = orig_cmap(np.linspace(0, 1, 5))
            discrete_colors[2] = mcolors.to_rgba("lightgray") # Set the exact middle bin to a neutral gray so it's visible against the brain template
            binned_cmap = mcolors.ListedColormap(discrete_colors, name="PRGn_5bins")

            # We need discrete normalization so the colorbar correctly reflects the bins
            bounds = np.linspace(0, 17000, 6) # 6 boundaries for 5 bins
            norm_binned = mcolors.BoundaryNorm(bounds, binned_cmap.N)

            out_name_brain2 = os.path.join(brain_dir, f"FWHM_brain_{mode}_{model.replace(' ', '_')}_PRGn_5bins.png")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            display = pu.plot_effect_glassbrain(
                subset,
                effect_col="FWHM",
                subjects=subjects_for_loading,
                cmap=binned_cmap,
                outfile=out_name_brain2,
                vmin=0, # vmin/vmax typically ignored by nilearn if discrete norm is parsed correctly but kept for safety
                vmax=17000,
                ax=ax2, 
                show=False,
                title=f"Binned FWHM - {model} ({mode}) | 5 Bins",
                node_kwargs=node_kwargs_outline,
            )
            # Natively replace the unbinned colorbar nilearn generates with a binned one
            if hasattr(display, '_colorbar_ax') and display._colorbar_ax is not None:
                cb_ax = display._colorbar_ax
                cb_ax.clear()
                plt.colorbar(plt.cm.ScalarMappable(norm=norm_binned, cmap=binned_cmap), cax=cb_ax, ticks=bounds)
            fig2.savefig(out_name_brain2)
            fig2.savefig(os.path.splitext(out_name_brain2)[0] + ".svg")
            plt.close(fig2)

            print(f"Saved centered and binned PRGn FWHM brainmaps for {model}")

        except Exception as e:
            print(f"Failed to plot brainmap for {model}: {e}")
            plt.close()


def export_roi_plots_all_models(datasets, lags_obj, roi, outdir, line_colors, mode):
    roi_datasets = []
    for name, df, line_label in datasets:
        if df is None or df.empty:
            continue
        roi_df = df[df["roi"] == roi]
        if roi_df.empty:
            continue
        roi_datasets.append((name, roi_df, line_label))

    if not roi_datasets:
        print(f"No data to plot for ROI {roi}")
        return

    outfile = os.path.join(outdir, f"{roi}_all-models_{mode}.pdf")
    export_electrodes_pdf_all_models(roi_datasets, lags_obj, outfile, line_colors)



def load_and_prep_data(res_d, context_paths, thresh_single):
    df_list_c = []
    df_list_p = []

    for _, base in context_paths:
        df_c = pu.load_res_add_roi_threshold(f"{base}comp.csv", thresh_single)
        df_p = pu.load_res_add_roi_threshold(f"{base}prod.csv", thresh_single)

        if "lag60-50" in base:
            df_c = trim_lags_to_window(df_c, 600, 1800)
            df_p = trim_lags_to_window(df_p, 600, 1800)

        df_list_c.append(df_c)
        df_list_p.append(df_p)

    word_list_c = [
        pu.load_res_add_roi_threshold(
            f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-var-partition_pca300_word-only_comp.csv",
            None,
        )
    ]
    for df in word_list_c:
        df["label3"] = "word"

    word_list_p = [
        pu.load_res_add_roi_threshold(
            f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-var-partition_pca300_word-only_prod.csv",
            None,
        )
    ]
    for df in word_list_p:
        df["label3"] = "word"

    return df_list_c, df_list_p, word_list_c, word_list_p


def prep_shared_datasets(df_list_c, df_list_p, word_list_c, word_list_p, titles):
    # Determine shared electrodes (union of robust electrodes across all models)
    shared_electrodes_c = shared_elecs_or_context_and_word(df_list_c, word_list_c, thresh=0.12)
    shared_electrodes_p = shared_elecs_or_context_and_word(df_list_p, word_list_p, thresh=0.12)

    # Filter datasets
    filtered_ctx_c = filter_by_shared_electrodes(df_list_c, shared_electrodes_c)
    filtered_ctx_p = filter_by_shared_electrodes(df_list_p, shared_electrodes_p)
    filtered_word_c = filter_by_shared_electrodes(word_list_c, shared_electrodes_c)
    filtered_word_p = filter_by_shared_electrodes(word_list_p, shared_electrodes_p)

    # Package into datasets [(name, df, label), ...]
    datasets_comp = []
    for title, df in zip(titles, filtered_ctx_c):
        df_joint = df[df["label3"] == "joint"]
        if not df_joint.empty:
            datasets_comp.append((title, df_joint, "joint"))
    if filtered_word_c:
        df_word = filtered_word_c[0][filtered_word_c[0]["label3"] == "word"]
        if not df_word.empty:
            datasets_comp.append(("Static word embedding", df_word, "word"))

    datasets_prod = []
    for title, df in zip(titles, filtered_ctx_p):
        df_joint = df[df["label3"] == "joint"]
        if not df_joint.empty:
            datasets_prod.append((title, df_joint, "joint"))
    if filtered_word_p:
        df_word = filtered_word_p[0][filtered_word_p[0]["label3"] == "word"]
        if not df_word.empty:
            datasets_prod.append(("Static word embedding", df_word, "word"))

    return datasets_comp, datasets_prod, shared_electrodes_c, shared_electrodes_p


def prep_individual_datasets(df_list, word_list, titles, thresh=0.1):
    datasets = []

    # Process Context models
    for title, df in zip(titles, df_list):
        df_joint = df[df["label3"] == "joint"]
        if df_joint.empty:
            continue

        # Filter electrodes > thresh for this specific model
        elecs = _electrodes_over_thresh(df, "joint", thresh)
        if not elecs:
            continue

        df_filt = df_joint[df_joint["electrode"].astype(str).isin(elecs)].copy()
        datasets.append((title, df_filt, "joint"))

    # Process Word model
    if word_list:
        df = word_list[0]
        df_word = df[df["label3"] == "word"]
        if not df_word.empty:
            elecs = _electrodes_over_thresh(df, "word", thresh)
            if elecs:
                df_filt = df_word[df_word["electrode"].astype(str).isin(elecs)].copy()
                datasets.append(("Static word embedding", df_filt, "word"))

    return datasets


def fdr_bh_correct(p_values):
    """Benjamini-Hochberg FDR correction for a 1D array-like of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    p_sorted = p[order]

    adjusted_sorted = p_sorted * n / (np.arange(1, n + 1))
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def stouffer_combine_pvalues(p_values):
    """Fast one-sided Stouffer combination of p-values."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.nan
    p = np.clip(p, 1e-300, 1 - 1e-16)
    z = np.sum(norm.isf(p)) / np.sqrt(p.size)
    return float(norm.sf(z))


def stouffers_roi_test(pvals_df, mode, roi, lags_obj, selected_subject_electrodes=None, fdr_correct=False, alpha=0.05):
    """
    Aggregate per-electrode p-values into ROI-level p-values per lag using Stouffer's method.
    Optionally applies FDR correction across lags.
    """
    if pvals_df is None or pvals_df.empty:
        return pd.DataFrame(columns=["lag_ms", "p_combined", "p_plot", "reject", "n_electrodes"])

    df = pvals_df[(pvals_df["mode"] == mode) & (pvals_df["roi"] == roi)].copy()
    if df.empty:
        return pd.DataFrame(columns=["lag_ms", "p_combined", "p_plot", "reject", "n_electrodes"])

    if selected_subject_electrodes:
        df["subject_electrode"] = df["subject"].astype(str) + "_" + df["electrode"].astype(str)
        df = df[df["subject_electrode"].isin(selected_subject_electrodes)].copy()

    if df.empty:
        return pd.DataFrame(columns=["lag_ms", "p_combined", "p_plot", "reject", "n_electrodes"])

    df["lag_ms_int"] = np.round(df["lag_ms"].astype(float)).astype(int)

    p_col = "p_value"

    test_lags = lags_obj.get("lags_test", lags_obj["lags_plt"])
    agg_rows = []
    for lag in test_lags:
        lag_int = int(np.round(lag))
        lag_slice = df[df["lag_ms_int"] == lag_int]
        p_slice = lag_slice[p_col].dropna().astype(float)

        if p_slice.empty:
            agg_rows.append({
                "lag_ms": lag_int,
                "p_combined": np.nan,
                "n_electrodes": 0,
            })
            continue

        p_combined = stouffer_combine_pvalues(p_slice.values)
        agg_rows.append({
            "lag_ms": lag_int,
            "p_combined": float(p_combined),
            "n_electrodes": int(len(p_slice.values)),
        })

    agg = pd.DataFrame(agg_rows)
    agg["p_plot"] = agg["p_combined"]
    
    if fdr_correct:
        valid_mask = agg["p_combined"].notna()
        agg.loc[valid_mask, "p_plot"] = fdr_bh_correct(agg.loc[valid_mask, "p_combined"].values)

    agg["reject"] = agg["p_plot"] < alpha
    return agg


def stouffers_line_test_from_context8_null(
    df_line,
    mode,
    lags_obj,
    null_res_d,
    fdr_correct=False,
    alpha=0.05,
    null_cache=None,
):
    """
    Recompute per-lag p-values for one plotted line using context8 null distributions,
    then combine across electrodes with Stouffer.
    """
    if df_line is None or df_line.empty:
        return pd.DataFrame(columns=["lag_ms", "p_combined", "p_plot", "reject", "n_electrodes"])

    if null_cache is None:
        null_cache = {}

    lag_pairs = []  # (lag_ms, lag_idx, col_name)
    lags_all = np.array(lags_obj["lags_all"])
    test_lags = lags_obj.get("lags_test", lags_obj["lags_plt"])
    
    for lag_val in test_lags:
        found_col = None
        lag_idx = None

        if lag_val in lags_all:
            lag_idx = int(np.where(lags_all == lag_val)[0][0])
            idx_str = str(lag_idx)
            if idx_str in df_line.columns:
                found_col = idx_str
            elif lag_idx in df_line.columns:
                found_col = lag_idx

        if found_col is None:
            lag_str = str(lag_val)
            lag_str_float = f"{float(lag_val):.1f}"
            if lag_str in df_line.columns:
                found_col = lag_str
            elif lag_str_float in df_line.columns:
                found_col = lag_str_float
            elif lag_val in df_line.columns:
                found_col = lag_val

        if found_col is not None and lag_idx is not None:
            lag_pairs.append((int(lag_val), lag_idx, found_col))

    if not lag_pairs:
        return pd.DataFrame(columns=["lag_ms", "p_combined", "p_plot", "reject", "n_electrodes"])

    pvals_by_lag = {lag_ms: [] for lag_ms, _, _ in lag_pairs}
    pvals_by_elec = {}

    for _, row in df_line.iterrows():
        sid = str(row["subject"])
        elec = str(row["electrode"])
        elec_key = f"{sid}_{elec}"
        pvals_by_elec.setdefault(elec_key, {})
        null_file = (
            f"{null_res_d}/ij-tfs-{sid}-mistral-stats-testing-1000perm-all-lags/"
            f"ij-200ms-{sid}/null_distributions/{sid}_{elec}_{mode}_null_perf.h5"
        )
        if not os.path.exists(null_file):
            continue

        if null_file not in null_cache:
            with h5py.File(null_file, "r") as h5f:
                null_corrs = h5f["null_corrs"][:]
                lag_indices = h5f["lag_indices"][:].astype(int)
            lag_to_col = {int(lag_idx): j for j, lag_idx in enumerate(lag_indices)}
            null_cache[null_file] = (null_corrs, lag_to_col)

        null_corrs, lag_to_col = null_cache[null_file]

        for lag_ms, lag_idx, col_name in lag_pairs:
            if lag_idx not in lag_to_col:
                continue
            try:
                obs_val = float(row[col_name])
            except Exception:
                continue
            null_vals = null_corrs[:, lag_to_col[lag_idx]]
            pval = (1.0 + np.sum(null_vals >= obs_val)) / (len(null_vals) + 1.0)
            pvals_by_elec[elec_key][lag_ms] = float(pval)

    # Always collect raw p-values per electrode
    for elec_key, elec_pmap in pvals_by_elec.items():
        for lag, p in elec_pmap.items():
            pvals_by_lag[lag].append(float(p))

    rows = []
    for lag_ms, _, _ in lag_pairs:
        p_slice = pvals_by_lag.get(lag_ms, [])
        if not p_slice:
            rows.append({"lag_ms": lag_ms, "p_combined": np.nan, "n_electrodes": 0})
            continue
        p_combined = stouffer_combine_pvalues(p_slice)
        rows.append({"lag_ms": lag_ms, "p_combined": float(p_combined), "n_electrodes": int(len(p_slice))})

    agg = pd.DataFrame(rows)
    agg["p_plot"] = agg["p_combined"]

    if fdr_correct:
        valid_mask = agg["p_combined"].notna()
        agg.loc[valid_mask, "p_plot"] = fdr_bh_correct(agg.loc[valid_mask, "p_combined"].values)

    agg["reject"] = agg["p_plot"] < alpha
    return agg


def _line_null_mean_distribution_at_lag(df_line, mode, lag_ms, lags_obj, null_res_d, null_cache=None):
    """Build null distribution of mean line value at one lag across selected electrodes."""
    if df_line is None or df_line.empty:
        return np.array([])
    if null_cache is None:
        null_cache = {}

    lags_all = np.array(lags_obj["lags_all"])
    if lag_ms not in lags_all:
        return np.array([])
    lag_idx = int(np.where(lags_all == lag_ms)[0][0])

    null_cols = []
    for _, row in df_line.iterrows():
        sid = str(row["subject"])
        elec = str(row["electrode"])
        null_file = (
            f"{null_res_d}/ij-tfs-{sid}-mistral-stats-testing-1000perm-all-lags/"
            f"ij-200ms-{sid}/null_distributions/{sid}_{elec}_{mode}_null_perf.h5"
        )
        if not os.path.exists(null_file):
            continue

        if null_file not in null_cache:
            with h5py.File(null_file, "r") as h5f:
                null_corrs = h5f["null_corrs"][:]
                lag_indices = h5f["lag_indices"][:].astype(int)
            lag_to_col = {int(li): j for j, li in enumerate(lag_indices)}
            null_cache[null_file] = (null_corrs, lag_to_col)

        null_corrs, lag_to_col = null_cache[null_file]
        if lag_idx not in lag_to_col:
            continue
        null_cols.append(null_corrs[:, lag_to_col[lag_idx]])

    if not null_cols:
        return np.array([])

    min_len = min(len(v) for v in null_cols)
    arr = np.stack([v[:min_len] for v in null_cols], axis=0)
    return arr.mean(axis=0)


def load_joint_from_electrode_dir(model_dir, mode):
    """
    Load per-electrode joint files from a model directory and convert to one-row-per-electrode format.
    Files are expected in the form: <sid>_<electrode>_<mode>_banded_joint.csv
    """
    pattern = os.path.join(model_dir, "**", f"*_{mode}_banded_joint.csv")
    files = sorted(glob.glob(pattern, recursive=True))

    rows = []
    file_re = re.compile(r"^(?P<sid>\d+)_(?P<electrode>.+)_(?P<mode>comp|prod)_banded_joint\.csv$")

    for file_path in files:
        file_name = os.path.basename(file_path)
        m = file_re.match(file_name)
        if not m:
            continue

        sid = int(m.group("sid"))
        electrode = m.group("electrode")

        mat = pd.read_csv(file_path, header=None)
        if mat.empty:
            continue

        vals = mat.mean(axis=0).astype(float).values
        row = {str(i): vals[i] for i in range(len(vals))}
        row.update({
            "subject": sid,
            "electrode": electrode,
            "label3": "joint",
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = pu.add_roi_label_to_results(df)
    df["subject"] = df["subject"].astype(str)
    df["electrode"] = df["electrode"].astype(str)
    return df


def load_multisubject_pvals_with_roi(res_d, sids):
    all_pvals = []
    for sid in sids:
        p_path = (
            f"{res_d}/ij-tfs-{sid}-mistral-stats-testing-1000perm-all-lags/"
            f"ij-200ms-{sid}/null_stats_summary/pvals_per_lag_electrode_mode.csv"
        )
        if not os.path.exists(p_path):
            print(f"Missing p-value file for sid {sid}: {p_path}")
            continue

        df = pd.read_csv(p_path)
        df = df.rename(columns={"sid": "subject"})
        df = pu.add_roi_label_to_results(df)
        df["subject"] = df["subject"].astype(str)
        df["electrode"] = df["electrode"].astype(str)
        all_pvals.append(df)

    if not all_pvals:
        return pd.DataFrame()

    return pd.concat(all_pvals, ignore_index=True)


def load_joint_from_subject_template(path_template, sids, mode):
    """
    Load and concatenate per-subject context model outputs from a %s path template.
    Example template:
    .../ij-tfs-%s-mistral-mistral_bandedRidge-..._cnxt8_deltasv2
    """
    all_dfs = []
    for sid in sids:
        model_dir = path_template % sid
        if not os.path.exists(model_dir):
            print(f"Missing context dir for sid {sid}: {model_dir}")
            continue
        df_sid = load_joint_from_electrode_dir(model_dir, mode=mode)
        if df_sid is not None and not df_sid.empty:
            all_dfs.append(df_sid)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def plot_roi_subset_lags(
    datasets,
    lags_obj,
    roi,
    mode,
    outfile,
    line_colors,
    ymax=0.18,
    sig_lags_by_name=None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False

    for i, (name, df, line_label) in enumerate(datasets):
        if df is None or df.empty:
            continue
        cols = _lag_columns_for_plot(df, lags_obj)
        if not cols:
            continue
        vals = df[cols].mean(axis=0)
        errs = df[cols].sem(axis=0)

        color = line_colors.get(name, "black")

        # Extract alpha if passed as a 4-tuple
        marker_alpha = 1.0
        if isinstance(color, tuple) and len(color) == 4:
            marker_alpha = color[3]

        # Use solid linestyle for all
        ls = "-"

        ax.plot(
            lags_obj["lags_plt"],
            vals,
            color=color,
            lw=2,
            linestyle=ls,
            label=name,
        )
        ax.fill_between(lags_obj["lags_plt"], vals - errs, vals + errs, color=color, alpha=0.1)

        if sig_lags_by_name is not None and name in sig_lags_by_name:
            sig_lags = sig_lags_by_name.get(name, np.array([]))
            if sig_lags is not None and len(sig_lags) > 0:
                lag_arr = np.array(lags_obj["lags_plt"], dtype=float)
                sig_x = [lag for lag in sig_lags if lag_arr.min() <= lag <= lag_arr.max()]
                if sig_x:
                    y_marker = 0.24 - 0.01 * i
                    sig_y = np.full(len(sig_x), y_marker)
                    ax.scatter(sig_x, sig_y, color=color, s=3, marker="o", zorder=6, alpha=marker_alpha)

        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ymin, ymax_val = ax.get_ylim()
    # Force ymax to include top significance markers
    ymax = 0.25
    if ymax is not None:
        ax.set_ylim(top=ymax, bottom=min(ymin, -0.025))
        ymin, ymax_val = ax.get_ylim()

    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")

    ax.set_xticks(lags_obj["lag_ticks"])
    ax.set_xticklabels(lags_obj["lag_tick_labels"])
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Correlation")
    ax.legend(loc="best", frameon=True, fontsize=10)
    mode_full = "Comprehension" if mode == "comp" else "Production"
    ax.set_title(f"{roi} {mode_full} (Subset)", fontsize=10)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    svg_out = os.path.splitext(outfile)[0] + ".svg"
    fig.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_roi_subset_plots(
    datasets,
    args,
    mode="comp",
    target_context="Context 16",
    null_res_d=None,
    sig_method="stouffer",
    output_suffix="",
):
    # Target subset
    target_names = [target_context, "Context 4", "Static word embedding"]
    
    # Build subset list ensuring order
    subset_datasets = []
    dataset_map = {name: (df, label) for name, df, label in datasets}
    
    for name in target_names:
        if name in dataset_map:
            subset_datasets.append((name, dataset_map[name][0], dataset_map[name][1]))
    
    if not subset_datasets:
        return

    # Colors: Blue with decreasing alpha
    # Base blue (C0): (0.1215, 0.4667, 0.7059)
    base_blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    
    roi_colors_map = {'MTG': '#018571', 'IFG': '#762a83', 'STG': '#67001f', 'postCG': '#543005'}

    lag_specs = [
        {
            "tag": "",
            "lags": {
                "lags_all": args.lags["lags_all"],
                "lags_test": np.arange(-30000, 30001, 50),
                "lags_plt": np.arange(-15000, 15001, 50),
                "lag_ticks": np.arange(-15000, 15001, 5000),
                "lag_tick_labels": np.arange(-15, 15.001, 5),
            },
        },
        {
            "tag": "_30s",
            "lags": {
                "lags_all": args.lags["lags_all"],
                "lags_test": np.arange(-30000, 30001, 50),
                "lags_plt": np.arange(-30000, 30001, 50),
                "lag_ticks": np.arange(-30000, 30001, 5000),
                "lag_tick_labels": np.arange(-30, 30.001, 5),
            },
        },
    ]

    out_dir = os.path.join(args.res_dir, "subset_plots")
    null_cache = {}
    
    if mode == "comp":
        selected_rois = ["MTG", "STG", "IFG"]
    else:
        selected_rois = ["IFG", "MTG", "postCG"]
        
    for roi in selected_rois:
        datasets_roi = []
        for name, df, label in subset_datasets:
            roi_df = df[df["roi"] == roi]
            if not roi_df.empty:
                datasets_roi.append((name, roi_df, label))

        if datasets_roi:
            # Use ROI-specific base color if available, otherwise default blue
            if roi in roi_colors_map:
                base_c = matplotlib.colors.to_rgb(roi_colors_map[roi])
            else:
                base_c = base_blue

            subset_colors = {
                target_context: base_c + (1.0,),
                "Context 4": base_c + (0.6,),
                "Static word embedding": base_c + (0.3,)
            }

            for lag_spec in lag_specs:
                lags_subset = lag_spec["lags"]
                lag_tag = lag_spec["tag"]
                roi_png = os.path.join(out_dir, f"{roi}_{mode}_subset{lag_tag}{output_suffix}.png")

                sig_cfgs = [
                    ("sig-fdr", True, 0.05),
                    ("sig-fdr-001", True, 0.01),
                    ("sig-fdr-0001", True, 0.001),
                ]

                for suffix, use_fdr, alpha in sig_cfgs:
                    sig_lags_by_name = None
                    line_sig_df = {}
                    if null_res_d is not None:
                        sig_lags_by_name = {}
                        for name, roi_df, _ in datasets_roi:
                            sig_df = stouffers_line_test_from_context8_null(
                                df_line=roi_df,
                                mode=mode,
                                lags_obj=lags_subset,
                                null_res_d=null_res_d,
                                fdr_correct=use_fdr,
                                alpha=alpha,
                                null_cache=null_cache,
                            )
                            line_sig_df[name] = sig_df
                            if not sig_df.empty:
                                sig_lags_by_name[name] = sig_df.loc[sig_df["reject"], "lag_ms"].astype(int).values

                    if (
                        roi == "IFG"
                        and mode == "prod"
                        and lag_tag == ""
                        and suffix in ("sig-uncorrected", "sig-fdr")
                        and sig_method == "stouffer"
                        and null_res_d is not None
                    ):
                        for name, roi_df, _ in datasets_roi:
                            sig_df = line_sig_df.get(name)
                            if sig_df is None or sig_df.empty:
                                continue
                            lag_row = sig_df[sig_df["lag_ms"] == -15000]
                            if lag_row.empty:
                                continue
                            observed_cols = _lag_columns_for_plot(roi_df, lags_subset)
                            lag_idx = int(np.where(np.array(lags_subset["lags_all"]) == -15000)[0][0])
                            lag_col = str(lag_idx) if str(lag_idx) in roi_df.columns else (lag_idx if lag_idx in roi_df.columns else None)
                            obs_mean = float(roi_df[lag_col].mean()) if lag_col is not None else np.nan

                            null_mean = _line_null_mean_distribution_at_lag(
                                roi_df,
                                mode,
                                -15000,
                                lags_subset,
                                null_res_d,
                                null_cache=null_cache,
                            )
                            crit_unc = float(np.quantile(null_mean, 0.95)) if null_mean.size else np.nan

                            p_raw = float(lag_row.iloc[0]["p_combined"])
                            p_plot = float(lag_row.iloc[0]["p_plot"])
                            pcrit = np.nan
                            if use_fdr:
                                rej = sig_df[sig_df["reject"]]
                                if not rej.empty:
                                    pcrit = float(rej["p_combined"].max())
                            crit_corr = float(np.quantile(null_mean, 1 - pcrit)) if (null_mean.size and not np.isnan(pcrit)) else np.nan

                            print(
                                f"[CRITICAL] ROI={roi} mode={mode} line={name} lag=-15.0s "
                                f"obs_mean={obs_mean:.6f} p_raw={p_raw:.6g} p_plot={p_plot:.6g} "
                                f"crit_unc_0.05={crit_unc:.6f} crit_corr={crit_corr:.6f}"
                            )

                    if suffix is None:
                        outfile = roi_png
                    else:
                        outfile = os.path.join(out_dir, f"{roi}_{mode}_subset{lag_tag}_{suffix}{output_suffix}.png")

                    plot_roi_subset_lags(
                        datasets_roi,
                        lags_subset,
                        roi,
                        mode,
                        outfile,
                        subset_colors,
                        sig_lags_by_name=sig_lags_by_name,
                    )


def plot_context16_all_rois(
    datasets,
    lags_obj,
    rois,
    mode,
    outfile,
    colors=None,
    linestyles=None,
    show_err=False,
    figsize=None,
    roi_colors=None,
    target_context="Context 16",
    pvals_df=None,
    sig_fdr=False,
    sig_alpha=0.05,
    sig_label=None,
    sig_method="stouffer",
):
    # Find target context dataset
    ctx16_entry = None
    for name, df, label in datasets:
        if name == target_context:
            ctx16_entry = (name, df, label)
            break
    
    if ctx16_entry is None:
        return

    name, df, label = ctx16_entry
    if df is None or df.empty:
        return

    fig, ax = plt.subplots(figsize=figsize if figsize else (10, 6))
    plotted = False
    
    rois_list = [r for r in rois if r != "All"]
    sig_levels = [0.22, 0.21, 0.20]

    for i, roi in enumerate(rois_list):
        roi_df = df[df["roi"] == roi]
        if roi_df.empty:
            continue
        n_elecs = roi_df[["subject", "electrode"]].drop_duplicates().shape[0]
            
        cols = _lag_columns_for_plot(roi_df, lags_obj)
        if not cols:
            continue
            
        vals = roi_df[cols].mean(axis=0)
        errs = roi_df[cols].sem(axis=0)
        
        # Determine color and linestyle
        if roi_colors and roi in roi_colors:
            color = roi_colors[roi]
        elif colors:
            color = colors[i % len(colors)]
        else:
             # Default Dark blue
             base_blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
             color = base_blue + (1.0,)

        marker_alpha = 1.0
        if isinstance(color, tuple) and len(color) == 4:
            marker_alpha = color[3]

        if linestyles:
            ls = linestyles[i % len(linestyles)]
        else:
             ls = "-"
        
        ax.plot(
            lags_obj["lags_plt"],
            vals,
            color=color,
            lw=2,
            linestyle=ls,
            label=f"{roi} n={n_elecs}",
            # alpha=0.9
        )
        if show_err:
            try:
                # Handle alpha if color is a tuple
                c_alpha = color
                if isinstance(c_alpha, tuple) and len(c_alpha) == 4:
                    # Use a lighter alpha for fill
                    c_alpha = (c_alpha[0], c_alpha[1], c_alpha[2], 0.1)
                elif isinstance(c_alpha, str) and c_alpha.startswith("#"):
                     # Convert hex to rgb for fill is cleaner but matplotlib handles string colors in fill_between ok usually
                     # Just use alpha kwarg
                     c_alpha = color
                
                ax.fill_between(lags_obj["lags_plt"], vals - errs, vals + errs, color=c_alpha, alpha=0.1)
            except:
                 ax.fill_between(lags_obj["lags_plt"], vals - errs, vals + errs, color=color, alpha=0.1)

        if target_context == "Context 8":
            if (pvals_df is not None) and (not pvals_df.empty):
                selected_subject_electrodes = {
                    f"{str(r['subject'])}_{str(r['electrode'])}"
                    for _, r in roi_df[["subject", "electrode"]].drop_duplicates().iterrows()
                }
                sig_df = stouffers_roi_test(
                    pvals_df=pvals_df,
                    mode=mode,
                    roi=roi,
                    lags_obj=lags_obj,
                    selected_subject_electrodes=selected_subject_electrodes,
                    fdr_correct=sig_fdr,
                    alpha=sig_alpha,
                )
            else:
                sig_df = pd.DataFrame()
            if not sig_df.empty:
                sig_lags = sig_df.loc[sig_df["reject"], "lag_ms"].astype(int).values
                if len(sig_lags) > 0:
                    lag_arr = np.array(lags_obj["lags_plt"], dtype=float)
                    sig_x = [lag for lag in sig_lags if lag_arr.min() <= lag <= lag_arr.max()]
                    if sig_x:
                        y_marker = sig_levels[i] if i < len(sig_levels) else (sig_levels[-1] - 0.01 * (i - len(sig_levels) + 1))
                        sig_y = np.full(len(sig_x), y_marker)
                        ax.scatter(sig_x, sig_y, color=color, s=3, marker="o", zorder=6, alpha=marker_alpha)
                 
        plotted = True

    if not plotted:
        plt.close(fig)
        return
     
    # Set ymax to include significance marker bands
    ax.set_ylim(top=0.23)
    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    
    ax.set_xticks(lags_obj["lag_ticks"])
    ax.set_xticklabels(lags_obj["lag_tick_labels"])
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Correlation")
    
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=1)
    
    mode_full = "Comprehension" if mode == "comp" else "Production"
    if target_context == "Context 8" and pvals_df is not None and not pvals_df.empty:
        if sig_label is None:
            sig_lbl = f"{'FDR' if sig_fdr else 'uncorrected'} p<{sig_alpha:g}"
        else:
            sig_lbl = sig_label
        method_lbl = "Stouffer"
        ax.set_title(f"{target_context} - All ROIs ({mode_full}, {method_lbl} {sig_lbl})", fontsize=12)
    else:
        ax.set_title(f"{target_context} - All ROIs ({mode_full})", fontsize=12)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    svg_out = os.path.splitext(outfile)[0] + ".svg"
    fig.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target_context} all-ROI plot to {outfile}")


def generate_context16_roi_plots(
    datasets,
    args,
    mode="comp",
    target_context="Context 16",
    pvals_df=None,
    sig_method="stouffer",
    output_suffix="",
):
    # Lags for all-roi context plots: full -30s to 30s testing, but -15s to 15s plotting
    lags_subset = {
        "lags_all": args.lags["lags_all"],
        "lags_test": np.arange(-30000, 30001, 50),
        "lags_plt": np.arange(-15000, 15001, 50),
        "lag_ticks": np.arange(-15000, 15001, 5000),
        "lag_tick_labels": np.arange(-15, 15.001, 5),
    }

    # Select ROIs based on mode
    if mode == "comp":
        selected_rois = ["MTG", "STG", "IFG"]
    else:
        selected_rois = ["IFG", "MTG", "postCG"]
    
    out_dir = os.path.join(args.res_dir, "subset_plots")
    
    roi_colors = {'MTG': '#018571', 'IFG': '#762a83', 'STG': '#67001f', 'postCG': '#543005'}
    context_slug = target_context.lower().replace(" ", "")

    plot_specs = [
        {
            "suffix": "_new_clrs",
            "kwargs": {"roi_colors": roi_colors, "linestyles": ["-"], "show_err": True, "figsize": (8, 4)},
        },
    ]

    if target_context == "Context 8" and pvals_df is not None and not pvals_df.empty:
        sig_modes = [
            ("fdr", True, 0.05, "FDR p<0.05"),
            ("fdr-001", True, 0.01, "FDR p<0.01"),
            ("fdr-0001", True, 0.001, "FDR p<0.001"),
        ]
    else:
        sig_modes = [(None, False, 0.05, None)]

    for spec in plot_specs:
        for sig_tag, sig_fdr, sig_alpha, sig_label in sig_modes:
            sig_suffix = f"_sig-{sig_tag}" if sig_tag is not None else ""
            outfile = os.path.join(out_dir, f"{context_slug}_all_rois_{mode}{spec['suffix']}{sig_suffix}{output_suffix}.png")
            plot_context16_all_rois(
                datasets,
                lags_subset,
                selected_rois,
                mode,
                outfile,
                target_context=target_context,
                pvals_df=(pvals_df if sig_tag is not None else None),
                sig_fdr=sig_fdr,
                sig_alpha=sig_alpha,
                sig_label=sig_label,
                sig_method=sig_method,
                **spec["kwargs"],
            )



def generate_roi_plots(datasets, args, line_colors, mode="comp", pvals_df=None):
    for roi in args.rois:
        datasets_roi = []
        for name, df, label in datasets:
            roi_df = df[df["roi"] == roi]
            if not roi_df.empty:
                datasets_roi.append((name, roi_df, label))

        if datasets_roi:
            roi_png = os.path.join(args.res_dir, f"{roi}_{mode}_all_models.png")
            plot_roi_all_models(
                datasets_roi,
                args.lags,
                roi,
                mode=mode,
                outfile=roi_png,
                line_colors=line_colors,
                pvals_df=pvals_df,
                sig_fdr=False,
                sig_alpha=0.05,
            )
            export_roi_plots_all_models(datasets, args.lags, roi, args.res_dir, line_colors, mode=mode)


def main():

    gen_lag_plots = False
    gen_subset_plots = True
    gen_fwhm_plots = True

    target_context = "Context 8"
    output_tag = "context8"


    args = Args()
    args.res_dir = os.path.join(args.res_dir, output_tag)
    os.makedirs(args.res_dir, exist_ok=True)

    res_d = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
    titles = ["Context 2", "Context 4", "Context 8", "Context 16", "Context 64"]
    thresh_single = None
    context_paths = [
        ("Context 2", f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context2_"),
        ("Context 4", f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context4_"),
        ("Context 8", f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context8_"),
        ("Context 16", f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_"),
        ("Context 64", f"{res_d}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context64_"),
    ]

    # New context-8 template containing per-electrode joint files for multiple patients
    context8_joint_template = (
        "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs/"
        "ij-tfs-%s-mistral-mistral_bandedRidge-lag30k-50-var-partition_pca300_all_load-splits_no-shift_cnxt8_deltasv2"
    )

    # P-values for ROI-level Stouffer test across subjects
    pval_sids = ["625", "676", "7170", "798"]
    pvals_df = load_multisubject_pvals_with_roi(res_d, pval_sids)


    # 1. Load Data
    df_list_c, df_list_p, word_list_c, word_list_p = load_and_prep_data(res_d, context_paths, thresh_single)
    
    if gen_lag_plots:

        # 2. Prep Shared Datasets for Lag Plots
        datasets_shared_c, datasets_shared_p, elecs_c, elecs_p = prep_shared_datasets(
            df_list_c, df_list_p, word_list_c, word_list_p, titles
        )

        print(f"Comprehension shared electrodes (OR>0.1 across context+word): {len(elecs_c)}")
        print(f"Production shared electrodes (OR>0.1 across context+word): {len(elecs_p)}")

        # 3. Colors
        palette = plt.get_cmap("tab10").colors
        context_colors = {title: palette[i % len(palette)] for i, title in enumerate(titles)}
        line_colors = {**context_colors}
        line_colors["Static word embedding"] = "black"

        # 4. Generate ROI Lag Plots (Shared Elecs)
        generate_roi_plots(datasets_shared_c, args, line_colors, mode="comp", pvals_df=pvals_df)
        generate_roi_plots(datasets_shared_p, args, line_colors, mode="prod", pvals_df=pvals_df)

    if gen_subset_plots:
        # Prep Shared Datasets if not already done (it is done in block above, but block above is conditional)
        # Assuming gen_lag_plots is True for now, or I should move the prep out.
        # But 'datasets_shared_c' are local to the if block? No, python scoping... 
        # Actually in python, variables defined in if block leak to outer scope if executed.
        # But if gen_lag_plots is False, this will fail.
        # I should check if datasets_shared_c exists or re-run prep.
        
        # Safer: Re-run prep or ensure it's available.
        # Since I'm editing main, let's look at the structure again.
        
        if 'datasets_shared_c' not in locals():
             datasets_shared_c, datasets_shared_p, _, _ = prep_shared_datasets(
                df_list_c, df_list_p, word_list_c, word_list_p, titles
            )

        # Replace target context with new context-8 electrode-directory data
        if target_context == "Context 8":
            context8_comp = load_joint_from_subject_template(context8_joint_template, pval_sids, mode="comp")
            context8_prod = load_joint_from_subject_template(context8_joint_template, pval_sids, mode="prod")

            if not context8_comp.empty:
                elecs_c = _electrodes_over_thresh(context8_comp, "joint", thresh=0.12)
                if elecs_c:
                    context8_comp = context8_comp[context8_comp["electrode"].astype(str).isin(elecs_c)].copy()

                datasets_shared_c = [
                    (name, context8_comp, label) if name == target_context else (name, df, label)
                    for name, df, label in datasets_shared_c
                ]

            if not context8_prod.empty:
                elecs_p = _electrodes_over_thresh(context8_prod, "joint", thresh=0.12)
                if elecs_p:
                    context8_prod = context8_prod[context8_prod["electrode"].astype(str).isin(elecs_p)].copy()

                datasets_shared_p = [
                    (name, context8_prod, label) if name == target_context else (name, df, label)
                    for name, df, label in datasets_shared_p
                ]
        
        generate_roi_subset_plots(
            datasets_shared_c,
            args,
            mode="comp",
            target_context=target_context,
            null_res_d=res_d,
            sig_method="stouffer",
        )
        generate_roi_subset_plots(
            datasets_shared_p,
            args,
            mode="prod",
            target_context=target_context,
            null_res_d=res_d,
            sig_method="stouffer",
        )

        generate_context16_roi_plots(
            datasets_shared_c,
            args,
            mode="comp",
            target_context=target_context,
            pvals_df=pvals_df,
            sig_method="stouffer",
        )
        generate_context16_roi_plots(
            datasets_shared_p,
            args,
            mode="prod",
            target_context=target_context,
            pvals_df=pvals_df,
            sig_method="stouffer",
        )

    if gen_fwhm_plots:
        # 5. Prep Individual Datasets for FWHM (Thresholded independently at 0.1)
        datasets_indiv_c = prep_individual_datasets(df_list_c, word_list_c, titles, thresh=0.12)
        datasets_indiv_p = prep_individual_datasets(df_list_p, word_list_p, titles, thresh=0.12)

        # 6. Generate FWHM Swarmplots and Brainmaps
        plot_fwhm_analysis(datasets_indiv_c, args.lags, args.rois, args.res_dir, mode="comp", model_filter=[target_context])
        plot_fwhm_analysis(datasets_indiv_p, args.lags, args.rois, args.res_dir, mode="prod", model_filter=[target_context])


if __name__ == "__main__":
    main()
