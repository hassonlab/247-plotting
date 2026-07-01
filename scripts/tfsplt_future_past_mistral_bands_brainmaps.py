import os
import re
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

import tfsplt_future_past_utils as pu

warnings.filterwarnings('ignore')


COMPONENT_SPECS = [
    ("u_fut", "Future/JointMax", "#0D8D00"),
    # ("u_word", "Word/JointMax", "#AC7000"),  # previous word color
    ("u_word", "Word/JointMax", "#ff7f0e"),
    ("u_pas", "Past/JointMax", "#A90008"),
]

DEFAULT_SUBJECTS = ["625", "676", "7170", "798"]
DEFAULT_LAGS_TO_PLOT = [-2000, -1000, -500, 0, 500, 1000, 2000]
MIRRORED_BAR_LAGS = list(range(-1000, 1100, 100))
MIRRORED_BAR_GROUP1 = list(range(0, 501, 100))
MIRRORED_BAR_GROUP2 = list(range(1000, 2001, 500))
MIRRORED_BAR_GROUP3 = list(range(5000, 25001, 10000))
MIRRORED_BAR_LAGS_30S = sorted(
    set(
        MIRRORED_BAR_GROUP1
        + MIRRORED_BAR_GROUP2
        + MIRRORED_BAR_GROUP3
        + [-x for x in MIRRORED_BAR_GROUP1 + MIRRORED_BAR_GROUP2 + MIRRORED_BAR_GROUP3 if x != 0]
    )
)
DEFAULT_JOINT_THRESH = 0.1
DEFAULT_RATIO_THRESH_SINGLE = 0.3
ALT_COMP_SPLIT_MS = 350
PEAK_LAG_MIN_MS = -1000
PEAK_LAG_MAX_MS = 1000
JOINT_ROW_COLOR = "#092C72"
JOINT_GRAY_FLOOR = 0.1

DEFAULT_RESULTS_DIR = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
DEFAULT_OUTPUT_DIR = (
    "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/roi_figures/"
    "future_past/mistral_variance_partitioning/brainmaps"
)

LABEL_TO_COMPONENT = {
    "joint": "joint",
    "sentence": "u_fut",
    "sentence2": "u_pas",
    "word": "u_word",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot mistral variance-partitioning brainmaps from per-electrode banded files. "
            "Loads from ij-200ms-<sid> directories, thresholds electrodes by joint performance, "
            "and saves ratio/non-ratio maps."
        )
    )
    parser.add_argument("--mode", choices=["prod", "comp", "both"], default="both")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--joint-thresh", type=float, default=DEFAULT_JOINT_THRESH)
    parser.add_argument("--ratio-thresh-single", type=float, default=DEFAULT_RATIO_THRESH_SINGLE)
    parser.add_argument("--lags", nargs="+", type=int, default=DEFAULT_LAGS_TO_PLOT)
    parser.add_argument("--ratio-vmin", type=float, default=0.0)
    parser.add_argument("--ratio-vmax", type=float, default=0.7)
    parser.add_argument("--abs-vmin", type=float, default=0.0)
    parser.add_argument("--abs-vmax", type=float, default=0.1)
    parser.add_argument("--joint-vmin", type=float, default=0.0)
    parser.add_argument("--joint-vmax", type=float, default=0.25)
    return parser.parse_args()


def _model_dir(results_dir, sid, model_name="all"):
    return (
        f"{results_dir}/ij-tfs-{sid}-mistral-mistral_bandedRidge-lag30k-50-var-partition_"
        f"pca300_{model_name}_load-splits_no-shift_cnxt8_deltasv2/ij-200ms-{sid}"
    )


def _numeric_lag_cols(df):
    return sorted([c for c in df.columns if c.isdigit()], key=lambda x: int(x))


def _load_label_from_electrode_dir(model_dir, mode, label):
    pattern = os.path.join(model_dir, f"*_{mode}_banded_{label}.csv")
    files = sorted(glob.glob(pattern))

    rows = []
    file_re = re.compile(rf"^(?P<sid>\d+)_(?P<electrode>.+)_(?P<mode>comp|prod)_banded_{label}\.csv$")

    for file_path in files:
        file_name = os.path.basename(file_path)
        match = file_re.match(file_name)
        if not match:
            continue

        sid = match.group("sid")
        electrode = match.group("electrode")

        mat = pd.read_csv(file_path, header=None)
        if mat.empty:
            continue

        vals = mat.mean(axis=0).astype(float).values
        row = {str(i): vals[i] for i in range(len(vals))}
        row.update({"subject": sid, "electrode": electrode, "label3": label})
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = pu.add_roi_label_to_results(df)
    df["subject"] = df["subject"].astype(str)
    df["electrode"] = df["electrode"].astype(str)
    return df


def _load_mode_data(results_dir, subjects, mode):
    loaded = {"joint": [], "u_fut": [], "u_pas": [], "u_word": []}

    for sid in subjects:
        model_dir = _model_dir(results_dir, sid, model_name="all")
        if not os.path.exists(model_dir):
            print(f"Missing dir for sid={sid}: {model_dir}")
            continue

        for label, comp_key in LABEL_TO_COMPONENT.items():
            df_sid = _load_label_from_electrode_dir(model_dir, mode, label)
            if df_sid.empty:
                print(f"No files loaded for sid={sid}, mode={mode}, label={label}")
                continue

            loaded[comp_key].append(df_sid)

    out = {}
    for comp_key, parts in loaded.items():
        if not parts:
            raise ValueError(f"No data loaded for component '{comp_key}' and mode '{mode}'.")
        out[comp_key] = pd.concat(parts, ignore_index=True)

    return out


def _available_lags_from_index_cols(index_cols):
    # Source lag indexing is 50 ms steps over [-30000, 30000].
    return [(-30000 + int(c) * 50) for c in index_cols]


def _nearest_available_lags(requested_lags, available_lags):
    return [min(available_lags, key=lambda x: abs(x - lag)) for lag in requested_lags]


def _split_allowed_lags(component_key, lags, split_ms=0, include_equal=True):
    if component_key == "u_fut":
        return [lag for lag in lags if lag <= split_ms] if include_equal else [lag for lag in lags if lag < split_ms]
    if component_key == "u_pas":
        return [lag for lag in lags if lag >= split_ms] if include_equal else [lag for lag in lags if lag > split_ms]
    return list(lags)


def _compute_component_values(mode_data, joint_thresh):
    key_cols = ["subject", "electrode", "roi"]
    joint_df = mode_data["joint"].copy()
    fut_df = mode_data["u_fut"].copy()
    pas_df = mode_data["u_pas"].copy()
    word_df = mode_data["u_word"].copy()

    common_idx_cols = set(_numeric_lag_cols(joint_df))
    common_idx_cols &= set(_numeric_lag_cols(fut_df))
    common_idx_cols &= set(_numeric_lag_cols(pas_df))
    common_idx_cols &= set(_numeric_lag_cols(word_df))
    common_idx_cols = sorted(common_idx_cols, key=lambda x: int(x))

    def renamed(df, suffix):
        renames = {c: f"{c}_{suffix}" for c in common_idx_cols}
        return df[key_cols + common_idx_cols].rename(columns=renames)

    merged = renamed(joint_df, "joint")
    merged = merged.merge(renamed(fut_df, "u_fut"), on=key_cols, how="inner")
    merged = merged.merge(renamed(pas_df, "u_pas"), on=key_cols, how="inner")
    merged = merged.merge(renamed(word_df, "u_word"), on=key_cols, how="inner")

    joint_cols = [f"{c}_joint" for c in common_idx_cols]
    keep_mask = merged[joint_cols].max(axis=1) > joint_thresh
    merged = merged[keep_mask].copy()

    if merged.empty:
        raise ValueError("No electrodes remained after joint thresholding.")

    lag_ms = _available_lags_from_index_cols(common_idx_cols)
    joint_max = merged[joint_cols].max(axis=1).replace(0, np.nan)

    ratio = {}
    abs_vals = {}
    base = merged[key_cols].copy()
    joint_abs_df = base.copy()

    for lag_idx, lag in zip(common_idx_cols, lag_ms):
        joint_abs_df[str(lag)] = np.maximum(pd.to_numeric(merged[f"{lag_idx}_joint"], errors="coerce"), 0)

    for component_key, _, _ in COMPONENT_SPECS:
        ratio_df = base.copy()
        abs_df = base.copy()

        for lag_idx, lag in zip(common_idx_cols, lag_ms):
            numerator = np.maximum(pd.to_numeric(merged[f"{lag_idx}_{component_key}"], errors="coerce"), 0)
            abs_df[str(lag)] = numerator
            ratio_df[str(lag)] = numerator / joint_max

        ratio[component_key] = ratio_df
        abs_vals[component_key] = abs_df

    joint_peak_df = base.copy()
    joint_peak_df["effect"] = joint_abs_df[[str(lag) for lag in lag_ms]].max(axis=1)
    joint_peak_df = joint_peak_df[np.isfinite(joint_peak_df["effect"])].copy()

    return ratio, abs_vals, joint_abs_df, joint_peak_df, lag_ms, len(merged)


def _effect_df(df_component, lag, ratio_thresh=0.0):
    lag_col = str(lag)
    out = df_component[["subject", "electrode", lag_col]].copy()
    out = out.rename(columns={lag_col: "effect"})
    return out[out["effect"] > ratio_thresh].copy()


def _effect_df_best_per_elec(df_component, allowed_lags, ratio_thresh=0.0):
    if not allowed_lags:
        raise ValueError("No allowed lags for per-electrode peak selection.")

    lag_cols = [str(lag) for lag in allowed_lags]
    vals = df_component[lag_cols].apply(pd.to_numeric, errors="coerce")

    out = df_component[["subject", "electrode"]].copy()
    out["effect"] = vals.max(axis=1)
    out = out[np.isfinite(out["effect"])].copy()
    return out[out["effect"] > ratio_thresh].copy()


def _pick_single_map_lags(ratios, all_lags, split_ms=0, include_equal=True):
    picked = {}

    def best_lag(component_key, allowed_lags):
        df_component = ratios[component_key]
        lag_means = {
            lag: np.nanmean(pd.to_numeric(df_component[str(lag)], errors="coerce"))
            for lag in allowed_lags
        }
        return max(lag_means, key=lag_means.get)

    picked["u_word"] = best_lag("u_word", all_lags)

    fut_lags = _split_allowed_lags("u_fut", all_lags, split_ms=split_ms, include_equal=include_equal)
    past_lags = _split_allowed_lags("u_pas", all_lags, split_ms=split_ms, include_equal=include_equal)
    if not fut_lags or not past_lags:
        raise ValueError("Split removed all valid lags for future or past.")

    picked["u_fut"] = best_lag("u_fut", fut_lags)
    picked["u_pas"] = best_lag("u_pas", past_lags)
    return picked


def _build_component_colormaps():
    out = {}
    for key, _, color in COMPONENT_SPECS:
        out[key] = mcolors.LinearSegmentedColormap.from_list(f"{key}_cmap", ["white", color], N=256)
    return out


def _gray_floor_cmap(base_cmap, vmin, vmax, gray_until, n=256):
    if gray_until <= vmin or vmax <= vmin:
        return base_cmap

    gray_until = min(gray_until, vmax)
    colors = base_cmap(np.linspace(0, 1, n))
    cutoff_idx = int(round((gray_until - vmin) / (vmax - vmin) * (n - 1)))
    cutoff_idx = max(0, min(cutoff_idx, n - 1))
    colors[: cutoff_idx + 1, :3] = np.array([0.7, 0.7, 0.7])
    return mcolors.ListedColormap(colors)


def _add_vertical_colorbar(fig, cax, cmap, vmin, vmax, label="Value"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
    cbar.set_label(label, rotation=270, labelpad=10)


def _plot_series(
    values_by_component,
    lags_to_plot,
    output_path,
    title,
    vmin,
    vmax,
    ratio_thresh=0.0,
    label_lags=None,
):
    """Plot series panels, using the provided lag labels for display."""
    if label_lags is None:
        label_lags = lags_to_plot
    cmaps = _build_component_colormaps()
    gray = mcolors.LinearSegmentedColormap.from_list("context_gray", ["white", "grey"], N=256)

    n_rows = len(COMPONENT_SPECS)
    n_cols = len(lags_to_plot) + 1
    fig = plt.figure(figsize=(3 * len(lags_to_plot) + 1, 3 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        hspace=0.3,
        wspace=0,
        width_ratios=[1] * len(lags_to_plot) + [0.1],
        left=0.05,
        right=0.95,
        top=0.93,
        bottom=0.05,
    )

    first_row_axes = []

    for row_idx, (component_key, _, _) in enumerate(COMPONENT_SPECS):
        df_component = values_by_component[component_key]

        for col_idx, lag in enumerate(lags_to_plot):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if row_idx == 0:
                first_row_axes.append(ax)

            if (component_key == "u_fut" and lag > 0) or (component_key == "u_pas" and lag < 0):
                cmap = gray
            else:
                cmap = cmaps[component_key]

            pu.plot_effect_glassbrain(
                _effect_df(df_component, lag, ratio_thresh=ratio_thresh),
                effect_col="effect",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                show=False,
                colorbar=False,
            )
            ax.margins(0)

        cbar_ax = fig.add_subplot(gs[row_idx, -1])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmaps[component_key]),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label("Value", rotation=270, labelpad=15)

    for lag, ax in zip(label_lags, first_row_axes):
        x_pos = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_pos, 0.95, f"{lag} ms", ha="center", va="center", fontsize=12, weight="bold")

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_abs_series_with_joint(
    abs_by_component,
    joint_abs_df,
    lags_to_plot,
    output_path,
    title,
    abs_vmin,
    abs_vmax,
    joint_vmin,
    joint_vmax,
    label_lags=None,
):
    """Plot absolute-value series panels, using the provided lag labels for display."""
    if label_lags is None:
        label_lags = lags_to_plot
    cmaps = _build_component_colormaps()
    cmaps["joint"] = mcolors.LinearSegmentedColormap.from_list("joint_cmap", ["white", JOINT_ROW_COLOR], N=256)
    gray = mcolors.LinearSegmentedColormap.from_list("context_gray", ["white", "grey"], N=256)

    row_specs = COMPONENT_SPECS + [("joint", "Joint", JOINT_ROW_COLOR)]
    n_rows = len(row_specs)
    n_cols = len(lags_to_plot) + 1
    fig = plt.figure(figsize=(3 * len(lags_to_plot) + 1, 3 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        hspace=0.3,
        wspace=0,
        width_ratios=[1] * len(lags_to_plot) + [0.1],
        left=0.05,
        right=0.95,
        top=0.93,
        bottom=0.05,
    )

    first_row_axes = []

    for row_idx, (component_key, _, _) in enumerate(row_specs):
        df_component = joint_abs_df if component_key == "joint" else abs_by_component[component_key]
        row_vmin = joint_vmin if component_key == "joint" else abs_vmin
        row_vmax = joint_vmax if component_key == "joint" else abs_vmax

        for col_idx, lag in enumerate(lags_to_plot):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if row_idx == 0:
                first_row_axes.append(ax)

            if component_key != "joint" and ((component_key == "u_fut" and lag > 0) or (component_key == "u_pas" and lag < 0)):
                cmap = gray
            else:
                cmap = cmaps[component_key]

            pu.plot_effect_glassbrain(
                _effect_df(df_component, lag, ratio_thresh=0.0),
                effect_col="effect",
                cmap=cmap,
                vmin=row_vmin,
                vmax=row_vmax,
                ax=ax,
                show=False,
                colorbar=False,
            )
            ax.margins(0)

        cbar_ax = fig.add_subplot(gs[row_idx, -1])
        norm = mcolors.Normalize(vmin=row_vmin, vmax=row_vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmaps[component_key]),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label("Value", rotation=270, labelpad=15)

    for lag, ax in zip(label_lags, first_row_axes):
        x_pos = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_pos, 0.95, f"{lag} ms", ha="center", va="center", fontsize=12, weight="bold")

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_single_global_peak(
    ratios,
    picked_lags,
    joint_peak_df,
    output_path,
    title,
    vmin,
    vmax,
    joint_vmin,
    joint_vmax,
    ratio_thresh_single=0.3,
):
    cmaps = _build_component_colormaps()
    cmaps["joint_peak"] = mcolors.LinearSegmentedColormap.from_list("joint_peak_cmap", ["white", JOINT_ROW_COLOR], N=256)

    panel_specs = COMPONENT_SPECS + [("joint_peak", "Joint Max", JOINT_ROW_COLOR)]
    n_panels = len(panel_specs)
    fig = plt.figure(figsize=(4.5 * n_panels, 4))
    gs = fig.add_gridspec(
        1,
        n_panels * 2,
        width_ratios=[1, 0.06] * n_panels,
        wspace=0.35,
        left=0.03,
        right=0.98,
        top=0.87,
        bottom=0.05,
    )

    for idx, (component_key, label, _) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, idx * 2])
        cax = fig.add_subplot(gs[0, idx * 2 + 1])
        if component_key == "joint_peak":
            plot_df = joint_peak_df[["subject", "electrode", "effect"]].copy()
            panel_vmin = joint_vmin
            panel_vmax = joint_vmax
            gray_floor = JOINT_GRAY_FLOOR
            panel_title = f"{label}\nMax across lags"
        else:
            lag = picked_lags[component_key]
            plot_df = _effect_df(ratios[component_key], lag, ratio_thresh=ratio_thresh_single)
            panel_vmin = vmin
            panel_vmax = vmax
            gray_floor = ratio_thresh_single
            panel_title = f"{label}\nLag {lag} ms"

        pu.plot_effect_glassbrain(
            plot_df,
            effect_col="effect",
            cmap=cmaps[component_key],
            vmin=panel_vmin,
            vmax=panel_vmax,
            ax=ax,
            show=False,
            colorbar=False,
            title=None,
        )
        cbar_cmap = _gray_floor_cmap(cmaps[component_key], vmin=panel_vmin, vmax=panel_vmax, gray_until=gray_floor)
        _add_vertical_colorbar(fig, cax, cbar_cmap, vmin=panel_vmin, vmax=panel_vmax, label="Ratio")
        ax.set_title(
            panel_title,
            fontsize=12,
            color="black",
            pad=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, boxstyle="square,pad=0.2"),
        )

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_single_best_per_elec(
    ratios,
    all_lags,
    joint_peak_df,
    output_path,
    title,
    vmin,
    vmax,
    joint_vmin,
    joint_vmax,
    split_ms=0,
    include_equal=True,
    ratio_thresh_single=0.3,
):
    cmaps = _build_component_colormaps()
    cmaps["joint_peak"] = mcolors.LinearSegmentedColormap.from_list("joint_peak_cmap", ["white", JOINT_ROW_COLOR], N=256)

    panel_specs = COMPONENT_SPECS + [("joint_peak", "Joint Max", JOINT_ROW_COLOR)]
    n_panels = len(panel_specs)
    fig = plt.figure(figsize=(4.5 * n_panels, 4))
    gs = fig.add_gridspec(
        1,
        n_panels * 2,
        width_ratios=[1, 0.06] * n_panels,
        wspace=0.35,
        left=0.03,
        right=0.98,
        top=0.87,
        bottom=0.05,
    )

    for idx, (component_key, label, _) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[0, idx * 2])
        cax = fig.add_subplot(gs[0, idx * 2 + 1])
        if component_key == "joint_peak":
            plot_df = joint_peak_df[["subject", "electrode", "effect"]].copy()
            panel_vmin = joint_vmin
            panel_vmax = joint_vmax
            gray_floor = JOINT_GRAY_FLOOR
            panel_title = f"{label}\nMax across lags"
        else:
            allowed_lags = _split_allowed_lags(component_key, all_lags, split_ms=split_ms, include_equal=include_equal)
            plot_df = _effect_df_best_per_elec(ratios[component_key], allowed_lags, ratio_thresh=ratio_thresh_single)
            panel_vmin = vmin
            panel_vmax = vmax
            gray_floor = ratio_thresh_single
            panel_title = f"{label}\nBest lag per electrode"

        pu.plot_effect_glassbrain(
            plot_df,
            effect_col="effect",
            cmap=cmaps[component_key],
            vmin=panel_vmin,
            vmax=panel_vmax,
            ax=ax,
            show=False,
            colorbar=False,
            title=None,
        )
        cbar_cmap = _gray_floor_cmap(cmaps[component_key], vmin=panel_vmin, vmax=panel_vmax, gray_until=gray_floor)
        _add_vertical_colorbar(fig, cax, cbar_cmap, vmin=panel_vmin, vmax=panel_vmax, label="Ratio")
        ax.set_title(
            panel_title,
            fontsize=12,
            color="black",
            pad=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, boxstyle="square,pad=0.2"),
        )

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _save_png_svg(plot_fn, stem_path, *args, **kwargs):
    for ext in ["png", "svg"]:
        plot_fn(*args, output_path=f"{stem_path}.{ext}", **kwargs)


def _unique_in_order(values):
    seen = set()
    out = []
    for val in values:
        if val in seen:
            continue
        out.append(val)
        seen.add(val)
    return out


def _is_group_boundary_gap(prev_lag, curr_lag):
    prev_abs = abs(prev_lag)
    curr_abs = abs(curr_lag)
    return (prev_abs, curr_abs) in {
        (500, 1000),
        (1000, 500),
        (2000, 5000),
        (5000, 2000),
    }


def _aggregate_component_by_roi(abs_by_component, lags):
    key_cols = ["subject", "electrode", "roi"]
    lag_cols = [str(lag) for lag in lags]
    merged = abs_by_component["u_word"][key_cols].copy()

    for component_key in ["u_word", "u_fut", "u_pas"]:
        df_comp = abs_by_component[component_key][key_cols + lag_cols].copy()
        rename_map = {str(lag): f"{lag}_{component_key}" for lag in lags}
        df_comp = df_comp.rename(columns=rename_map)
        merged = merged.merge(df_comp, on=key_cols, how="inner")

    value_cols = [f"{lag}_{comp}" for lag in lags for comp in ["u_word", "u_fut", "u_pas"]]
    for col in value_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    roi_counts = merged["roi"].value_counts()
    roi_df = merged.groupby("roi")[value_cols].mean().reset_index()
    return roi_df, roi_counts


def _plot_mirrored_mistral_bars(
    prod_abs_by_component,
    comp_abs_by_component,
    lags,
    output_dir,
    min_count_per_mode=5,
    grey_out=True,
    show_gap_ellipses=False,
    filename_suffix="",
    gap_spacing_fraction=0.0,
    use_group_boundary_spacing=False,
    max_y=0.3,
    show_xticks_top=False,
    show_xticks_bottom=True,
    show_xticks_bottom_top=False,
    legend_top_right=False,
    masked_gray_color="#f4f4f4",
    tick_labels_in_seconds=False,
    show_inner_axis_arrow=False,
    tick_label_fontsize=11,
    tick_label_rotation=0,
    bottom_top_tick_pad=8,
    bottom_top_tick_label_y=1.045,
    bottom_top_tick_xoffset_pts=0.0,
    gap_marker_y=0.04,
    onset_label_y=-0.12,
    arrow_end_offset=0.65,
    subplot_hspace=0.25,
):
    os.makedirs(output_dir, exist_ok=True)

    prod_roi, prod_counts = _aggregate_component_by_roi(prod_abs_by_component, lags)
    comp_roi, comp_counts = _aggregate_component_by_roi(comp_abs_by_component, lags)

    common_rois = [
        roi for roi in set(prod_roi["roi"]) & set(comp_roi["roi"])
        if prod_counts.get(roi, 0) > min_count_per_mode and comp_counts.get(roi, 0) > min_count_per_mode
    ]
    common_rois = sorted(common_rois)

    if not common_rois:
        print("No common ROIs passed mirrored-bar count threshold; skipping mirrored bars.")
        return

    orange, green, red, grey = "#ff7f0e", "#2ca02c", "#d62728", masked_gray_color
    style_map = {
        "u_word": {"color": orange, "label": "Word"},
        "u_fut": {"color": green, "label": "Future (Sentence)"},
        "u_pas": {"color": red, "label": "Past (Sentence2)"},
    }
    plot_categories = ["u_word", "u_fut", "u_pas"]

    def _style_bar_container(container, colors):
        for patch, bar_color in zip(container.patches, colors):
            if patch.get_height() <= 0:
                # Avoid zero-height outlines that can create a solid+dash double top edge.
                patch.set_edgecolor("none")
                patch.set_linewidth(0)
                continue
            if bar_color == grey:
                patch.set_edgecolor("#8e8e8e")
                patch.set_linewidth(0.6)
                patch.set_linestyle((0, (2, 2)))
            else:
                patch.set_edgecolor("white")
                patch.set_linewidth(0.2)
                patch.set_linestyle("solid")

    for roi in common_rois:
        p_row = prod_roi[prod_roi["roi"] == roi]
        c_row = comp_roi[comp_roi["roi"] == roi]
        if p_row.empty or c_row.empty:
            continue

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"hspace": subplot_hspace},
        )

        bar_width = 0.8

        gap_after_idx = set()
        if gap_spacing_fraction > 0 and len(lags) > 1:
            x_vals = [0.0]
            extra_step = bar_width * gap_spacing_fraction
            for i in range(1, len(lags)):
                step = 1.0
                if use_group_boundary_spacing:
                    needs_gap = _is_group_boundary_gap(lags[i - 1], lags[i])
                else:
                    needs_gap = abs(lags[i] - lags[i - 1]) > 100

                if needs_gap:
                    step += extra_step
                    gap_after_idx.add(i - 1)
                x_vals.append(x_vals[-1] + step)
            x_pos = np.array(x_vals)
        else:
            x_pos = np.arange(len(lags))

        bottom_p = np.zeros(len(lags))
        bottom_c = np.zeros(len(lags))

        # Draw in layers so masked future (gray) is above past (red) for positive lags.
        draw_layers = [
            "u_word",
            "u_fut_nonpos",
            "u_pas",
            "u_fut_pos",
        ]

        for layer in draw_layers:
            p_vals, c_vals, color_list = [], [], []

            for lag in lags:
                p_word = max(0, p_row[f"{lag}_u_word"].values[0])
                c_word = max(0, c_row[f"{lag}_u_word"].values[0])
                p_fut = max(0, p_row[f"{lag}_u_fut"].values[0])
                c_fut = max(0, c_row[f"{lag}_u_fut"].values[0])
                p_pas = max(0, p_row[f"{lag}_u_pas"].values[0])
                c_pas = max(0, c_row[f"{lag}_u_pas"].values[0])

                if layer == "u_word":
                    pv, cv = p_word, c_word
                    col = orange
                elif layer == "u_fut_nonpos":
                    if lag < 0:
                        pv, cv = p_fut, c_fut
                        col = green
                    else:
                        pv, cv = 0.0, 0.0
                        col = green
                elif layer == "u_pas":
                    pv, cv = p_pas, c_pas
                    col = red
                    if grey_out and lag <= 0:
                        col = grey
                else:  # u_fut_pos
                    if lag >= 0:
                        pv, cv = p_fut, c_fut
                        col = grey if grey_out else green
                    else:
                        pv, cv = 0.0, 0.0
                        col = grey if grey_out else green

                p_vals.append(pv)
                c_vals.append(cv)
                color_list.append(col)

            top_container = ax_top.bar(
                x_pos,
                p_vals,
                width=bar_width,
                bottom=bottom_p,
                color=color_list,
            )
            bot_container = ax_bot.bar(
                x_pos,
                c_vals,
                width=bar_width,
                bottom=bottom_c,
                color=color_list,
            )
            _style_bar_container(top_container, color_list)
            _style_bar_container(bot_container, color_list)

            bottom_p += np.array(p_vals)
            bottom_c += np.array(c_vals)

        ax_top.set_ylim(0, max_y)
        ax_bot.set_ylim(max_y, 0)

        for ax in [ax_top, ax_bot]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.tick_params(left=True, labelleft=True, axis="y", labelsize=tick_label_fontsize)

        def _lag_tick_label(lag_ms):
            if not tick_labels_in_seconds:
                return f"{lag_ms}"
            lag_s = lag_ms / 1000.0
            if abs(lag_s - int(lag_s)) < 1e-9:
                return f"{int(lag_s)}"
            return f"{lag_s:g}"

        labels = [_lag_tick_label(l) for l in lags]
        tick_ha = "right" if tick_label_rotation else "center"
        ax_top.set_xticks(x_pos)
        ax_top.set_xticklabels(labels, rotation=tick_label_rotation, ha=tick_ha, rotation_mode="anchor")
        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels(labels, rotation=tick_label_rotation, ha=tick_ha, rotation_mode="anchor")

        if show_xticks_top:
            # Top-axis bottom ticks sit between the mirrored panels.
            ax_top.tick_params(axis="x", which="major", bottom=True, top=False, labelbottom=True, pad=1, labelsize=tick_label_fontsize)
        else:
            ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # For sparse lag selections, annotate large temporal gaps between adjacent bars.
        if show_gap_ellipses and len(lags) > 1:
            for i in range(len(lags) - 1):
                show_gap = i in gap_after_idx
                if show_gap:
                    mid_x = (x_pos[i] + x_pos[i + 1]) / 2
                    ax_top.text(
                        mid_x,
                        gap_marker_y,
                        "//",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                    )
                    ax_bot.text(
                        mid_x,
                        gap_marker_y,
                        "//",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                    )

        ax_bot.tick_params(
            axis="x",
            which="major",
            bottom=show_xticks_bottom,
            top=show_xticks_bottom_top,
            labelbottom=show_xticks_bottom,
            labeltop=False,
            pad=bottom_top_tick_pad if show_xticks_bottom_top else 1,
            labelsize=tick_label_fontsize,
        )
        # Draw custom lower-panel top labels at exact x positions to avoid side-dependent drift.
        if show_xticks_bottom_top:
            label_transform = ax_bot.get_xaxis_transform()
            if bottom_top_tick_xoffset_pts != 0.0:
                label_transform = mtransforms.offset_copy(
                    label_transform,
                    fig=fig,
                    x=bottom_top_tick_xoffset_pts,
                    y=0.0,
                    units="points",
                )
            for x, label in zip(x_pos, labels):
                ax_bot.text(
                    x,
                    bottom_top_tick_label_y,
                    label,
                    transform=label_transform,
                    ha=tick_ha,
                    va="bottom",
                    rotation=tick_label_rotation,
                    rotation_mode="anchor",
                    fontsize=tick_label_fontsize,
                    color="black",
                    clip_on=False,
                )
        ax_bot.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)

        if show_inner_axis_arrow:
            ax_top.annotate(
                "",
                xy=(x_pos[-1] + arrow_end_offset, 0.0),
                xytext=(x_pos[0] - arrow_end_offset, 0.0),
                xycoords="data",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
                annotation_clip=False,
            )
            ax_bot.annotate(
                "",
                xy=(x_pos[-1] + arrow_end_offset, 0.0),
                xytext=(x_pos[0] - arrow_end_offset, 0.0),
                xycoords="data",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
                annotation_clip=False,
            )
            ax_top.text(
                (x_pos[0] + x_pos[-1]) / 2,
                onset_label_y,
                "onset lag (s)",
                transform=ax_top.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                color="black",
            )

        fig.suptitle(
            f"{roi}\n(n={prod_counts[roi]} Prod, n={comp_counts[roi]} Comp)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        legend_handles = [
            Patch(facecolor=style_map[cat]["color"], label=style_map[cat]["label"])
            for cat in plot_categories
        ]
        if grey_out:
            legend_handles.append(
                Patch(facecolor=grey, edgecolor="#8e8e8e", linestyle=(0, (2, 2)), linewidth=0.6, label="Context (masked)")
            )
        if legend_top_right:
            ax_top.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
        else:
            ax_top.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

        clean_name = roi.replace("/", "_").replace(" ", "_")
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        for ext in ["png", "svg"]:
            fig.savefig(
                os.path.join(output_dir, f"mirrored_mistral{suffix}_{clean_name}.{ext}"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def _run_mirrored_bars_from_mode_data(comp_data, prod_data, args):
    comp_ratios, comp_abs, _, _, comp_lags, _ = _compute_component_values(comp_data, args.joint_thresh)
    prod_ratios, prod_abs, _, _, prod_lags, _ = _compute_component_values(prod_data, args.joint_thresh)

    del comp_ratios
    del prod_ratios

    common_lags = sorted(set(comp_lags) & set(prod_lags))
    if not common_lags:
        print("No common lags found between comp and prod for mirrored bars.")
        return

    # Match banded mirrored bars: fixed lag window from -1000 to 1000 in 100 ms steps.
    lags_to_plot = _unique_in_order(_nearest_available_lags(MIRRORED_BAR_LAGS, common_lags))
    out_dir = os.path.join(args.output_dir, "mirrored_bars")
    _plot_mirrored_mistral_bars(
        prod_abs,
        comp_abs,
        lags_to_plot,
        out_dir,
        min_count_per_mode=5,
        grey_out=True,
        show_gap_ellipses=False,
        filename_suffix="",
        gap_spacing_fraction=0.0,
        use_group_boundary_spacing=False,
        max_y=0.3,
        show_xticks_top=False,
        show_xticks_bottom=True,
        show_xticks_bottom_top=False,
        legend_top_right=False,
        masked_gray_color="#f4f4f4",
        tick_labels_in_seconds=False,
        show_inner_axis_arrow=False,
        tick_label_fontsize=11,
        arrow_end_offset=0.65,
        subplot_hspace=0.25,
    )

    # Extended mirrored bars over [-30s, 30s] with sparse lag set and ellipsis markers for large gaps.
    lags_30s = _unique_in_order(_nearest_available_lags(MIRRORED_BAR_LAGS_30S, common_lags))
    _plot_mirrored_mistral_bars(
        prod_abs,
        comp_abs,
        lags_30s,
        out_dir,
        min_count_per_mode=5,
        grey_out=True,
        show_gap_ellipses=True,
        filename_suffix="30s",
        gap_spacing_fraction=1.0,
        use_group_boundary_spacing=True,
        max_y=0.25,
        show_xticks_top=True,
        show_xticks_bottom=False,
        show_xticks_bottom_top=True,
        legend_top_right=True,
        masked_gray_color="#fcfcfc",
        tick_labels_in_seconds=True,
        show_inner_axis_arrow=True,
        tick_label_fontsize=10,
        tick_label_rotation=0,
        bottom_top_tick_pad=1,
        bottom_top_tick_label_y=1.09,
        bottom_top_tick_xoffset_pts=0.0,
        gap_marker_y=0.04,
        onset_label_y=-0.13,
        arrow_end_offset=0.8,
        subplot_hspace=0.48,
    )


def _run_mode(mode_name, mode_data, args, split_ms=0, include_equal=True, suffix="", series_label_shift_ms=0):
    ratios, abs_vals, joint_abs, joint_peak, all_lags, n_elecs = _compute_component_values(mode_data, args.joint_thresh)
    lags_to_plot = _nearest_available_lags(args.lags, all_lags)
    peak_lags = [lag for lag in all_lags if PEAK_LAG_MIN_MS <= lag <= PEAK_LAG_MAX_MS]
    series_lags = [lag + series_label_shift_ms for lag in lags_to_plot]

    mode_tag = "comprehension" if mode_name == "comp" else "production"
    split_tag = f"split{split_ms}ms_{'incl' if include_equal else 'strict'}"
    base_tag = f"{mode_tag}_jt{int(round(args.joint_thresh * 100)):02d}{suffix}"

    print(f"[{mode_tag}] n electrodes after joint threshold {args.joint_thresh}: {n_elecs}")
    print(f"[{mode_tag}] lags_to_plot: {lags_to_plot}")

    # 1) Peak ratio maps from all lags (global + per-electrode variants)
    picked = _pick_single_map_lags(ratios, peak_lags, split_ms=split_ms, include_equal=include_equal)
    _save_png_svg(
        _plot_single_global_peak,
        os.path.join(args.output_dir, f"{base_tag}_peak-ratio-global_{split_tag}"),
        ratios,
        picked,
        joint_peak,
        title=f"{mode_tag.capitalize()}: Peak Ratio Map (Global Best Lag)",
        vmin=args.ratio_vmin,
        vmax=args.ratio_vmax,
        joint_vmin=args.joint_vmin,
        joint_vmax=args.joint_vmax,
        ratio_thresh_single=args.ratio_thresh_single,
    )
    _save_png_svg(
        _plot_single_best_per_elec,
        os.path.join(args.output_dir, f"{base_tag}_peak-ratio-best-per-elec_{split_tag}"),
        ratios,
        peak_lags,
        joint_peak,
        title=f"{mode_tag.capitalize()}: Peak Ratio Map (Best Lag Per Electrode)",
        vmin=args.ratio_vmin,
        vmax=args.ratio_vmax,
        joint_vmin=args.joint_vmin,
        joint_vmax=args.joint_vmax,
        split_ms=split_ms,
        include_equal=include_equal,
        ratio_thresh_single=args.ratio_thresh_single,
    )

    # 3) Timeseries ratio maps for selected lags, no ratio threshold
    _save_png_svg(
        _plot_series,
        os.path.join(args.output_dir, f"{base_tag}_series-ratio_selected-lags"),
        ratios,
        series_lags,
        title=f"{mode_tag.capitalize()}: Ratio Brain Maps Across Selected Lags",
        vmin=args.ratio_vmin,
        vmax=args.ratio_vmax,
        ratio_thresh=0.0,
    )

    # 4) Timeseries non-ratio absolute-value maps for selected lags
    _save_png_svg(
        _plot_abs_series_with_joint,
        os.path.join(args.output_dir, f"{base_tag}_series-abs_selected-lags"),
        abs_vals,
        joint_abs,
        series_lags,
        title=f"{mode_tag.capitalize()}: Absolute Component Maps Across Selected Lags",
        abs_vmin=args.abs_vmin,
        abs_vmax=args.abs_vmax,
        joint_vmin=args.joint_vmin,
        joint_vmax=args.joint_vmax,
    )


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    comp_data = None
    prod_data = None

    if args.mode in ["comp", "both"]:
        comp_data = _load_mode_data(args.results_dir, args.subjects, mode="comp")

        # Primary comprehension run: split at 0 ms
        _run_mode("comp", comp_data, args, split_ms=0, include_equal=True, suffix="")

        # 2) Control run for comprehension: strict split at 350 ms
        _run_mode(
            "comp",
            comp_data,
            args,
            split_ms=ALT_COMP_SPLIT_MS,
            include_equal=False,
            suffix="_control350ms",
            series_label_shift_ms=ALT_COMP_SPLIT_MS,
        )

    if args.mode in ["prod", "both"]:
        prod_data = _load_mode_data(args.results_dir, args.subjects, mode="prod")
        _run_mode("prod", prod_data, args, split_ms=0, include_equal=True, suffix="")

    if args.mode == "both" and comp_data is not None and prod_data is not None:
        _run_mirrored_bars_from_mode_data(comp_data, prod_data, args)


if __name__ == "__main__":
    main()
