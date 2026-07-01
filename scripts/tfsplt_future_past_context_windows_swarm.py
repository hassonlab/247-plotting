import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tfsplt_future_past_utils as pu


RES_D = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
OUT_DIR = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/roi_figures/future_past/context_windows_swarm"

MODE_SPECS = {
    "comp": [
        ("Static", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-var-partition_pca300_word-only_comp.csv", None),
        ("Context 2", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context2_comp.csv", "joint"),
        ("Context 4", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context4_comp.csv", "joint"),
        ("Context 8", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context8_comp.csv", "joint"),
        ("Context 16", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_comp.csv", "joint"),
        ("Context 64", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context64_comp.csv", "joint"),
    ],
    "prod": [
        ("Static", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-var-partition_pca300_word-only_prod.csv", None),
        ("Context 2", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context2_prod.csv", "joint"),
        ("Context 4", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context4_prod.csv", "joint"),
        ("Context 8", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context8_prod.csv", "joint"),
        ("Context 16", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag60-50-all-static_future_past-reph-translate-control_pca300_drop-short_mistral_prod.csv", "joint"),
        ("Context 64", f"{RES_D}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-all-static_future_past-noreph_pca300_30s_context64_prod.csv", "joint"),
    ],
}

CONTEXT_ORDER = ["Static", "Context 2", "Context 4", "Context 8", "Context 16", "Context 64"]
LAG_STEP_MS = 50

# Future/Past colors from tfsplt_future_past_variance_partitioning.py
FUTURE_PAST_COLORS = {
    "future": "#2ca02d",  # Green
    "past": "#d62829",    # Red
}


def get_center_index(max_lag):
    if max_lag == 1200:
        return 600
    if max_lag == 2400:
        return 1200
    return max_lag // 2


def ms_to_index_offset(ms):
    return int(ms / LAG_STEP_MS)


def remove_outliers(df, value_col):
    """Remove outliers using IQR method (1.5*IQR)."""
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)].copy()


def load_max_joint(path, label):
    df = pu.load_res_add_roi_threshold(path, None)
    if label is not None and "label3" in df.columns and df["label3"].notna().any():
        df = df[df["label3"] == label].copy()
    df = df.drop_duplicates(["subject", "electrode"])

    lag_cols = sorted([int(col) for col in df.columns if str(col).isdigit()])
    center = get_center_index(max(lag_cols))
    offset = ms_to_index_offset(1000)
    window_cols = [col for col in lag_cols if center - offset <= col <= center + offset]

    df["max_joint"] = df[[str(col) for col in window_cols]].max(axis=1)
    return df[["subject", "electrode", "max_joint"]].copy()


def plot_mode(mode):
    frames = {name: load_max_joint(path, label) for name, path, label in MODE_SPECS[mode]}
    counts = {}
    for df in frames.values():
        keys = df.loc[df["max_joint"] > 0.1, "subject"].astype(str) + "_" + df.loc[df["max_joint"] > 0.1, "electrode"].astype(str)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    selected = {key for key, count in counts.items() if count >= 2}

    plot_rows = []
    for context in CONTEXT_ORDER:
        df = frames[context]
        electrode_key = df["subject"].astype(str) + "_" + df["electrode"].astype(str)
        plot_rows.append(df[electrode_key.isin(selected)].assign(context=context)[["context", "max_joint"]])

    plot_df = pd.concat(plot_rows, ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 4))
    sns.violinplot(data=plot_df, x="context", y="max_joint", order=CONTEXT_ORDER, inner="quart", cut=0)
    plt.ylabel("max_joint")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{mode}_context_windows_max_joint_swarm.png"), dpi=300)
    plt.close()


def make_plot_df(frames):
    counts = {}
    for df in frames.values():
        keys = df.loc[df["max_joint"] > 0.1, "subject"].astype(str) + "_" + df.loc[df["max_joint"] > 0.1, "electrode"].astype(str)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    selected = {key for key, count in counts.items() if count >= 2}

    plot_rows = []
    for context in CONTEXT_ORDER:
        df = frames[context]
        electrode_key = df["subject"].astype(str) + "_" + df["electrode"].astype(str)
        plot_rows.append(df[electrode_key.isin(selected)].assign(context=context)[["context", "max_joint"]])

    return pd.concat(plot_rows, ignore_index=True)


def plot_modes():
    comp_df = make_plot_df({name: load_max_joint(path, label) for name, path, label in MODE_SPECS["comp"]})
    prod_df = make_plot_df({name: load_max_joint(path, label) for name, path, label in MODE_SPECS["prod"]})

    comp_df["mode"] = "comp"
    prod_df["mode"] = "prod"
    plot_df = pd.concat([comp_df, prod_df], ignore_index=True)
    plot_df = remove_outliers(plot_df, "max_joint")

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=plot_df, x="context", y="max_joint", hue="mode", split=True, order=CONTEXT_ORDER, inner="quart", cut=0, gap=0.1)
    plt.ylabel("max_joint")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comp_prod_context_windows_max_joint_violin.png"), dpi=300)
    plt.close()


def get_selected_electrodes(mode):
    """Get electrodes appearing with max_joint > 0.1 in >= 2 analyses."""
    frames = {name: load_max_joint(path, label) for name, path, label in MODE_SPECS[mode]}
    counts = {}
    for df in frames.values():
        keys = df.loc[df["max_joint"] > 0.1, "subject"].astype(str) + "_" + df.loc[df["max_joint"] > 0.1, "electrode"].astype(str)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= 2}


def load_full_correlations(path, label, selected_electrodes):
    """Load full correlation profiles for selected electrodes."""
    df = pu.load_res_add_roi_threshold(path, None)
    if label is not None and "label3" in df.columns and df["label3"].notna().any():
        df = df[df["label3"] == label].copy()
    df = df.drop_duplicates(["subject", "electrode"])
    
    electrode_key = df["subject"].astype(str) + "_" + df["electrode"].astype(str)
    df = df[electrode_key.isin(selected_electrodes)].copy()
    
    # Get lag columns
    lag_cols = sorted([int(col) for col in df.columns if str(col).isdigit()])
    corr_cols = [str(col) for col in lag_cols]
    
    return df, lag_cols, corr_cols


def compute_auc(df, lag_cols, lag_range):
    """Compute AUC for each electrode in specified lag range."""
    center = get_center_index(max(lag_cols))

    if lag_range is None:
        window_lags = lag_cols
    else:
        offset = ms_to_index_offset(lag_range)
        window_lags = [lag for lag in lag_cols if center - offset <= lag <= center + offset]

    return df[[str(lag) for lag in window_lags]].sum(axis=1).tolist()


def plot_auc_comp_prod(mode_specs, lag_range, title_suffix):
    """Plot AUC comparison between comp and prod."""
    comp_selected = get_selected_electrodes("comp")
    prod_selected = get_selected_electrodes("prod")
    
    plot_rows = []
    
    for context_name, comp_path, comp_label in mode_specs["comp"]:
        df_comp, lag_cols_comp, _ = load_full_correlations(comp_path, comp_label, comp_selected)
        if len(df_comp) > 0:
            auc_vals = compute_auc(df_comp, lag_cols_comp, lag_range)
            for auc in auc_vals:
                plot_rows.append({"context": context_name, "auc": auc, "mode": "comp"})
    
    for context_name, prod_path, prod_label in mode_specs["prod"]:
        df_prod, lag_cols_prod, _ = load_full_correlations(prod_path, prod_label, prod_selected)
        if len(df_prod) > 0:
            auc_vals = compute_auc(df_prod, lag_cols_prod, lag_range)
            for auc in auc_vals:
                plot_rows.append({"context": context_name, "auc": auc, "mode": "prod"})
    
    plot_df = pd.DataFrame(plot_rows)
    plot_df = remove_outliers(plot_df, "auc")

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=plot_df, x="context", y="auc", hue="mode", split=True, order=CONTEXT_ORDER, inner="quart", cut=0, gap=0.1)
    plt.ylabel("AUC")
    plt.xlabel("")
    plt.title(f"AUC {title_suffix}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"comp_prod_context_windows_auc_{title_suffix.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def get_future_past_peaks(mode):
    """Get peak values in future (-1000:0) and past (0:1000) lag windows for selected electrodes.
    Uses 'sentence' label for future and 'sentence2' label for past."""
    selected = get_selected_electrodes(mode)
    plot_rows = []
    
    # Skip static (index 0)
    for context_name, path, _ in MODE_SPECS[mode][1:]:
        # Load future with 'sentence' label
        df_future, lag_cols_future, _ = load_full_correlations(path, "sentence", selected)
        # Load past with 'sentence2' label
        df_past, lag_cols_past, _ = load_full_correlations(path, "sentence2", selected)

        center_f = get_center_index(max(lag_cols_future)) if len(df_future) > 0 else None
        center_p = get_center_index(max(lag_cols_past)) if len(df_past) > 0 else None
        offset = ms_to_index_offset(1000)

        # Compute future peaks
        if len(df_future) > 0 and center_f is not None:
            future_cols = [str(lag) for lag in lag_cols_future if center_f - offset <= lag <= center_f]
            future_peak = df_future[future_cols].max(axis=1) if future_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "peak": value, "direction": "future"} for value in future_peak.tolist())

        # Compute past peaks
        if len(df_past) > 0 and center_p is not None:
            past_cols = [str(lag) for lag in lag_cols_past if center_p <= lag <= center_p + offset]
            past_peak = df_past[past_cols].max(axis=1) if past_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "peak": value, "direction": "past"} for value in past_peak.tolist())
    
    return pd.DataFrame(plot_rows)


def plot_future_past_peaks(mode):
    """Plot future vs past peak comparison for given mode."""
    plot_df = get_future_past_peaks(mode)
    plot_df = remove_outliers(plot_df, "peak")
    
    # Filter to context windows (skip Static)
    context_order_no_static = CONTEXT_ORDER[1:]
    plot_df = plot_df[plot_df["context"].isin(context_order_no_static)]
    
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=plot_df, x="context", y="peak", hue="direction", split=True, 
                   order=context_order_no_static, inner="quart", cut=0, gap=0.1, palette=FUTURE_PAST_COLORS)
    plt.ylabel("Peak correlation")
    plt.xlabel("")
    plt.title(f"{mode.upper()}: Future (-1000:0ms) vs Past (0:1000ms)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{mode}_future_past_peaks.png"), dpi=300)
    plt.close()


def get_future_past_auc(mode):
    """Get AUC values in future (-1000:0) and past (0:1000) lag windows for selected electrodes.
    Uses 'sentence' label for future and 'sentence2' label for past."""
    selected = get_selected_electrodes(mode)
    plot_rows = []

    for context_name, path, _ in MODE_SPECS[mode][1:]:
        # Load future with 'sentence' label
        df_future, lag_cols_future, _ = load_full_correlations(path, "sentence", selected)
        # Load past with 'sentence2' label
        df_past, lag_cols_past, _ = load_full_correlations(path, "sentence2", selected)

        center_f = get_center_index(max(lag_cols_future)) if len(df_future) > 0 else None
        center_p = get_center_index(max(lag_cols_past)) if len(df_past) > 0 else None
        offset = ms_to_index_offset(1000)

        # Compute future AUC
        if len(df_future) > 0 and center_f is not None:
            future_cols = [str(lag) for lag in lag_cols_future if center_f - offset <= lag <= center_f]
            future_auc = df_future[future_cols].sum(axis=1) if future_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "auc": value, "direction": "future"} for value in future_auc.tolist())

        # Compute past AUC
        if len(df_past) > 0 and center_p is not None:
            past_cols = [str(lag) for lag in lag_cols_past if center_p <= lag <= center_p + offset]
            past_auc = df_past[past_cols].sum(axis=1) if past_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "auc": value, "direction": "past"} for value in past_auc.tolist())

    return pd.DataFrame(plot_rows)


def plot_future_past_auc(mode):
    """Plot future vs past AUC comparison for given mode."""
    plot_df = get_future_past_auc(mode)
    plot_df = remove_outliers(plot_df, "auc")

    context_order_no_static = CONTEXT_ORDER[1:]
    plot_df = plot_df[plot_df["context"].isin(context_order_no_static)]

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=plot_df, x="context", y="auc", hue="direction", split=True,
                   order=context_order_no_static, inner="quart", cut=0, gap=0.1, palette=FUTURE_PAST_COLORS)
    plt.ylabel("AUC")
    plt.xlabel("")
    plt.title(f"{mode.upper()}: Future (-1000:0ms) vs Past (0:1000ms)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{mode}_future_past_auc.png"), dpi=300)
    plt.close()


def get_future_past_auc_all_lags(mode):
    """Get AUC values using all lags: future = all negative lags, past = all positive lags.
    Uses 'sentence' label for future and 'sentence2' label for past."""
    selected = get_selected_electrodes(mode)
    plot_rows = []

    for context_name, path, _ in MODE_SPECS[mode][1:]:
        # Load future with 'sentence' label
        df_future, lag_cols_future, _ = load_full_correlations(path, "sentence", selected)
        # Load past with 'sentence2' label
        df_past, lag_cols_past, _ = load_full_correlations(path, "sentence2", selected)

        center_f = get_center_index(max(lag_cols_future)) if len(df_future) > 0 else None
        center_p = get_center_index(max(lag_cols_past)) if len(df_past) > 0 else None

        # Compute future AUC (all negative lags)
        if len(df_future) > 0 and center_f is not None:
            future_cols = [str(lag) for lag in lag_cols_future if lag < center_f]
            future_auc = df_future[future_cols].sum(axis=1) if future_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "auc": value, "direction": "future"} for value in future_auc.tolist())

        # Compute past AUC (all positive lags)
        if len(df_past) > 0 and center_p is not None:
            past_cols = [str(lag) for lag in lag_cols_past if lag > center_p]
            past_auc = df_past[past_cols].sum(axis=1) if past_cols else pd.Series(dtype=float)
            plot_rows.extend({"context": context_name, "auc": value, "direction": "past"} for value in past_auc.tolist())

    return pd.DataFrame(plot_rows)


def plot_future_past_auc_all_lags(mode):
    """Plot future vs past AUC using all lags."""
    plot_df = get_future_past_auc_all_lags(mode)
    plot_df = remove_outliers(plot_df, "auc")

    context_order_no_static = CONTEXT_ORDER[1:]
    plot_df = plot_df[plot_df["context"].isin(context_order_no_static)]

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=plot_df, x="context", y="auc", hue="direction", split=True,
                   order=context_order_no_static, inner="quart", cut=0, gap=0.1, palette=FUTURE_PAST_COLORS)
    plt.ylabel("AUC")
    plt.xlabel("")
    plt.title(f"{mode.upper()}: Future (all negative lags) vs Past (all positive lags)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{mode}_future_past_auc_all_lags.png"), dpi=300)
    plt.close()


def main():
    plot_modes()
    plot_auc_comp_prod(MODE_SPECS, None, "all lags")
    plot_auc_comp_prod(MODE_SPECS, 5000, "-5000:5000 ms")
    plot_auc_comp_prod(MODE_SPECS, 1000, "-1000:1000 ms")
    plot_future_past_peaks("comp")
    plot_future_past_peaks("prod")
    plot_future_past_auc("comp")
    plot_future_past_auc("prod")
    plot_future_past_auc_all_lags("comp")
    plot_future_past_auc_all_lags("prod")


if __name__ == "__main__":
    main()