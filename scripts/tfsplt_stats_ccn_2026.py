import glob
import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats


"""
Standalone CCN 2026 stats summary for mistral variance partitioning.

What this script does:
1) Loads true encoding results from
   {RESULTS_DIR}/ij-tfs-{sid}-mistral-mistral_bandedRidge-lag30k-50-var-partition_...
   and averages across all bands (CSV rows) for each electrode/mode/component.
2) Loads permutation null distributions from
   ij-tfs-{sid}-mistral-stats-testing-1000perm-all-lags/ij-200ms-{sid}/null_distributions*
3) Builds per-electrode stats from lag-window averages in -250:250 ms:
   - true_window_r: mean true r over the window
   - null_window_r: mean null r over the window per permutation
   - p-value from null >= true
   - 95th percentile critical value of null_window_r
4) Applies BH-FDR separately for comprehension and production.
5) Produces requested summaries:
   - # electrodes passing FDR + mean critical value
   - # electrodes with true_window_r > 0.1
   - best lag for comp/prod (global lag with highest mean joint performance)
   - mean per-electrode best lag performance among electrodes passing r > 0.1
   - comp vs prod comparisons for future/past in both raw and ratio form, where:
       future uses lags <= -100 ms
       past uses lags >= 100 ms
     ratio uses max(component, 0) / max(joint, 0) as in brainmap code.

Outputs are written to OUTPUT_DIR as CSV files and echoed in terminal.
"""


SUBJECTS = ["625", "676", "7170", "798"]

RESULTS_DIR = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
OUTPUT_DIR = (
    "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/misc/ccn_2026_stats"
)

MODEL_DIR_TEMPLATE = (
    "ij-tfs-{sid}-mistral-mistral_bandedRidge-lag30k-50-var-partition_"
    "pca300_all_load-splits_no-shift_cnxt8_deltasv2/ij-200ms-{sid}"
)
STATS_DIR_TEMPLATE = "ij-tfs-{sid}-mistral-stats-testing-1000perm-all-lags/ij-200ms-{sid}"

LABEL_MAP = {
    "joint": "joint",
    "future": "sentence",
    "past": "sentence2",
    "word": "word",
}

NULL_SUBDIR_CANDIDATES = ["null_distributions2", "null_distributions"]

LAGS_MS = np.arange(-30000, 30001, 50)
WINDOW_MIN_MS = -250
WINDOW_MAX_MS = 250
FUTURE_MAX_LAG_MS = -100
PAST_MIN_LAG_MS = 100

ALPHA = 0.05
CRITICAL_QUANTILE = 0.95
R_THRESH = 0.1

TRUE_FILE_RE = re.compile(
    r"^(?P<sid>\d+)_(?P<electrode>.+)_(?P<mode>comp|prod)_banded_(?P<label>joint|sentence|sentence2|word)\.csv$"
)
NULL_FILE_RE = re.compile(r"^(?P<sid>\d+)_(?P<electrode>.+)_(?P<mode>comp|prod)_null_perf\.h5$")


def _bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _safe_max(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmax(arr))


def _resolve_null_subdir(stats_sid_dir):
    for sub in NULL_SUBDIR_CANDIDATES:
        candidate = os.path.join(stats_sid_dir, sub)
        if os.path.isdir(candidate):
            h5s = glob.glob(os.path.join(candidate, "*_null_perf.h5"))
            if h5s:
                return candidate

    fallback = sorted(glob.glob(os.path.join(stats_sid_dir, "null_distributions*")))
    for candidate in fallback:
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "*_null_perf.h5")):
            return candidate
    return None


def _load_true_series_all_bands():
    data = {
        "comp": {"joint": {}, "future": {}, "past": {}, "word": {}},
        "prod": {"joint": {}, "future": {}, "past": {}, "word": {}},
    }
    missing_model_dirs = []

    for sid in SUBJECTS:
        model_dir = os.path.join(RESULTS_DIR, MODEL_DIR_TEMPLATE.format(sid=sid))
        if not os.path.isdir(model_dir):
            missing_model_dirs.append(model_dir)
            continue

        for component_key, label in LABEL_MAP.items():
            for mode in ["comp", "prod"]:
                pattern = os.path.join(model_dir, f"*_{mode}_banded_{label}.csv")
                for file_path in sorted(glob.glob(pattern)):
                    name = os.path.basename(file_path)
                    m = TRUE_FILE_RE.match(name)
                    if not m:
                        continue

                    mat = pd.read_csv(file_path, header=None).to_numpy(dtype=float)
                    if mat.ndim == 1:
                        series = mat.astype(float)
                    else:
                        # "All bands" here means averaging across rows of banded output.
                        series = np.nanmean(mat, axis=0)

                    key = (m.group("sid"), m.group("electrode"))
                    data[mode][component_key][key] = series

    return data, missing_model_dirs


def _align_true_series_to_null(true_series, null_num_lags, lag_indices):
    true_series = np.asarray(true_series, dtype=float)

    if lag_indices is not None:
        lag_indices = np.asarray(lag_indices, dtype=int)
        if true_series.shape[0] == LAGS_MS.shape[0] and lag_indices.shape[0] == null_num_lags:
            return true_series[lag_indices], LAGS_MS[lag_indices]

    if true_series.shape[0] == null_num_lags:
        if lag_indices is not None and lag_indices.shape[0] == null_num_lags and np.max(lag_indices) < LAGS_MS.shape[0]:
            return true_series, LAGS_MS[lag_indices]
        return true_series, LAGS_MS[:null_num_lags]

    return None, None


def _compute_null_window_stats_per_electrode(true_joint_by_mode):
    records = []
    missing_true = []
    missing_stats_dirs = []

    for sid in SUBJECTS:
        stats_sid_dir = os.path.join(RESULTS_DIR, STATS_DIR_TEMPLATE.format(sid=sid))
        if not os.path.isdir(stats_sid_dir):
            missing_stats_dirs.append(stats_sid_dir)
            continue

        null_dir = _resolve_null_subdir(stats_sid_dir)
        if null_dir is None:
            print(f"No null distributions found for sid={sid} in {stats_sid_dir}")
            continue

        for null_file in sorted(glob.glob(os.path.join(null_dir, "*_null_perf.h5"))):
            name = os.path.basename(null_file)
            m = NULL_FILE_RE.match(name)
            if not m:
                continue

            mode = m.group("mode")
            key = (m.group("sid"), m.group("electrode"))

            true_series = true_joint_by_mode[mode].get(key)
            if true_series is None:
                missing_true.append((sid, key[1], mode, null_file))
                continue

            with h5py.File(null_file, "r") as h5f:
                null_corrs = h5f["null_corrs"][:].astype(float)
                lag_indices = h5f["lag_indices"][:] if "lag_indices" in h5f else None

            aligned_true, aligned_lags = _align_true_series_to_null(
                true_series=true_series,
                null_num_lags=int(null_corrs.shape[1]),
                lag_indices=lag_indices,
            )
            if aligned_true is None:
                continue

            win_mask = (aligned_lags >= WINDOW_MIN_MS) & (aligned_lags <= WINDOW_MAX_MS)
            if not np.any(win_mask):
                continue

            true_win = float(np.nanmean(aligned_true[win_mask]))
            null_win = np.nanmean(null_corrs[:, win_mask], axis=1)
            null_win = null_win[np.isfinite(null_win)]
            if null_win.size == 0:
                continue

            p_val = (np.sum(null_win >= true_win) + 1.0) / (null_win.size + 1.0)
            critical = float(np.quantile(null_win, CRITICAL_QUANTILE))

            best_idx = int(np.nanargmax(aligned_true))
            best_lag = int(aligned_lags[best_idx])
            best_val = float(aligned_true[best_idx])

            records.append(
                {
                    "sid": key[0],
                    "electrode": key[1],
                    "mode": mode,
                    "true_window_r": true_win,
                    "null_window_mean_r": float(np.nanmean(null_win)),
                    "critical_value": critical,
                    "p_value": float(p_val),
                    "best_lag_ms_per_electrode": best_lag,
                    "best_lag_r_per_electrode": best_val,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No electrode records were computed. Check paths and file naming.")

    df["p_value_fdr"] = np.nan
    df["passes_fdr"] = False

    for mode in ["comp", "prod"]:
        idx = df["mode"] == mode
        corrected = _bh_fdr(df.loc[idx, "p_value"].to_numpy(dtype=float))
        df.loc[idx, "p_value_fdr"] = corrected
        df.loc[idx, "passes_fdr"] = corrected <= ALPHA

    df["passes_r_thresh"] = df["true_window_r"] > R_THRESH
    return df, missing_true, missing_stats_dirs


def _mode_global_best_lag(true_joint_by_mode, mode, keep_keys=None):
    series_dict = true_joint_by_mode[mode]
    keys = sorted(series_dict.keys())
    if keep_keys is not None:
        keep_keys = set(keep_keys)
        keys = [k for k in keys if k in keep_keys]
    if not keys:
        return np.nan, np.nan

    mat = np.vstack([np.asarray(series_dict[k], dtype=float) for k in keys])
    lag_means = np.nanmean(mat, axis=0)
    if lag_means.size != LAGS_MS.size:
        return np.nan, np.nan

    best_idx = int(np.nanargmax(lag_means))
    return int(LAGS_MS[best_idx]), float(lag_means[best_idx])


def _mean_best_per_electrode(true_joint_by_mode, mode, keep_keys):
    vals = []
    for key in sorted(set(keep_keys)):
        series = true_joint_by_mode[mode].get(key)
        if series is None:
            continue
        vals.append(_safe_max(series))
    return _safe_mean(vals)


def _paired_comp_prod_future_past(true_data):
    comp = true_data["comp"]
    prod = true_data["prod"]

    required = ["joint", "future", "past"]
    common_keys = set(comp["joint"]).intersection(prod["joint"])
    for k in required:
        common_keys &= set(comp[k])
        common_keys &= set(prod[k])

    if not common_keys:
        return pd.DataFrame()

    fut_mask = LAGS_MS <= FUTURE_MAX_LAG_MS
    past_mask = LAGS_MS >= PAST_MIN_LAG_MS

    rows = []
    for sid, electrode in sorted(common_keys):
        row = {"sid": sid, "electrode": electrode}
        for mode, mode_data in [("comp", comp), ("prod", prod)]:
            j = np.maximum(np.asarray(mode_data["joint"][(sid, electrode)], dtype=float), 0)
            f = np.maximum(np.asarray(mode_data["future"][(sid, electrode)], dtype=float), 0)
            p = np.maximum(np.asarray(mode_data["past"][(sid, electrode)], dtype=float), 0)

            j_max = _safe_max(j)
            fut_raw = _safe_max(f[fut_mask])
            past_raw = _safe_max(p[past_mask])

            if np.isfinite(j_max) and j_max > 0:
                fut_ratio = fut_raw / j_max
                past_ratio = past_raw / j_max
            else:
                fut_ratio = np.nan
                past_ratio = np.nan

            row[f"{mode}_future_raw"] = fut_raw
            row[f"{mode}_past_raw"] = past_raw
            row[f"{mode}_future_ratio"] = fut_ratio
            row[f"{mode}_past_ratio"] = past_ratio

        rows.append(row)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df

    def summarize_pair(metric_name, comp_col, prod_col):
        x = pair_df[comp_col].to_numpy(dtype=float)
        y = pair_df[prod_col].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]

        out = {
            "metric": metric_name,
            "n_pairs": int(x.size),
            "comp_mean": _safe_mean(x),
            "prod_mean": _safe_mean(y),
            "mean_diff_comp_minus_prod": _safe_mean(x - y),
            "paired_t_stat": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
        }

        if x.size >= 2:
            t_stat, t_p = stats.ttest_rel(x, y, nan_policy="omit")
            out["paired_t_stat"] = float(t_stat)
            out["paired_t_p"] = float(t_p)

        if x.size >= 1:
            try:
                w_stat, w_p = stats.wilcoxon(x, y)
                out["wilcoxon_stat"] = float(w_stat)
                out["wilcoxon_p"] = float(w_p)
            except ValueError:
                pass

        return out

    summary_rows = [
        summarize_pair("future_ratio", "comp_future_ratio", "prod_future_ratio"),
        summarize_pair("past_ratio", "comp_past_ratio", "prod_past_ratio"),
        summarize_pair("future_raw", "comp_future_raw", "prod_future_raw"),
        summarize_pair("past_raw", "comp_past_raw", "prod_past_raw"),
    ]
    return pd.DataFrame(summary_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    true_data, missing_model_dirs = _load_true_series_all_bands()
    true_joint_by_mode = {
        "comp": true_data["comp"]["joint"],
        "prod": true_data["prod"]["joint"],
    }

    electrode_df, missing_true, missing_stats_dirs = _compute_null_window_stats_per_electrode(true_joint_by_mode)

    mode_rows = []
    for mode in ["comp", "prod"]:
        mode_df = electrode_df[electrode_df["mode"] == mode].copy()
        keep_thresh = set(mode_df.loc[mode_df["passes_r_thresh"], ["sid", "electrode"]].itertuples(index=False, name=None))
        if not keep_thresh:
            keep_thresh = set(mode_df[["sid", "electrode"]].itertuples(index=False, name=None))

        best_lag_ms, best_lag_mean_r = _mode_global_best_lag(
            true_joint_by_mode=true_joint_by_mode,
            mode=mode,
            keep_keys=keep_thresh,
        )
        mean_best_per_elec = _mean_best_per_electrode(
            true_joint_by_mode=true_joint_by_mode,
            mode=mode,
            keep_keys=keep_thresh,
        )

        mode_rows.append(
            {
                "mode": mode,
                "n_electrodes_tested": int(mode_df.shape[0]),
                "n_pass_fdr": int(mode_df["passes_fdr"].sum()),
                "mean_critical_value_all": _safe_mean(mode_df["critical_value"]),
                "mean_critical_value_fdr_pass": _safe_mean(mode_df.loc[mode_df["passes_fdr"], "critical_value"]),
                "n_pass_r_gt_0p1": int(mode_df["passes_r_thresh"].sum()),
                "best_lag_ms": best_lag_ms,
                "best_lag_mean_r": best_lag_mean_r,
                "mean_best_lag_perf_per_electrode": mean_best_per_elec,
                "mean_window_r": _safe_mean(mode_df.loc[mode_df["passes_r_thresh"], "true_window_r"]),
            }
        )

    mode_summary_df = pd.DataFrame(mode_rows)
    comparison_df = _paired_comp_prod_future_past(true_data)

    electrode_out = os.path.join(OUTPUT_DIR, "ccn2026_electrode_stats_window_-250_250.csv")
    mode_out = os.path.join(OUTPUT_DIR, "ccn2026_mode_summary.csv")
    comparison_out = os.path.join(OUTPUT_DIR, "ccn2026_future_past_comp_vs_prod.csv")

    electrode_df.sort_values(["mode", "sid", "electrode"]).to_csv(electrode_out, index=False)
    mode_summary_df.to_csv(mode_out, index=False)
    comparison_df.to_csv(comparison_out, index=False)

    if missing_model_dirs:
        pd.DataFrame({"missing_model_dir": missing_model_dirs}).to_csv(
            os.path.join(OUTPUT_DIR, "ccn2026_missing_model_dirs.csv"),
            index=False,
        )

    if missing_stats_dirs:
        pd.DataFrame({"missing_stats_dir": missing_stats_dirs}).to_csv(
            os.path.join(OUTPUT_DIR, "ccn2026_missing_stats_dirs.csv"),
            index=False,
        )

    if missing_true:
        pd.DataFrame(
            missing_true,
            columns=["sid", "electrode", "mode", "null_file"],
        ).to_csv(
            os.path.join(OUTPUT_DIR, "ccn2026_missing_true_for_null.csv"),
            index=False,
        )

    print("Saved:")
    print(f"- {electrode_out}")
    print(f"- {mode_out}")
    print(f"- {comparison_out}")

    print("\nRequested headline summary:")
    for _, row in mode_summary_df.iterrows():
        print(
            f"[{row['mode']}] "
            f"FDR pass={int(row['n_pass_fdr'])}, "
            f"mean critical (all)={row['mean_critical_value_all']:.5f}, "
            f"r>0.1 pass={int(row['n_pass_r_gt_0p1'])}, "
            f"best lag={row['best_lag_ms']} ms, "
            f"mean best-per-electrode={row['mean_best_lag_perf_per_electrode']:.5f}"
        )

    if not comparison_df.empty:
        print("\nComp vs Prod comparisons (future/past, ratio/raw):")
        print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()