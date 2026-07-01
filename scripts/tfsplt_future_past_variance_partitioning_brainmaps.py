import os
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tfsplt_future_past_utils as pu
from tfsplt_future_past_variance_partitioning import run_variance_partitioning_pipeline

warnings.filterwarnings('ignore')


COMPONENT_SPECS = [
    ("u_fut", "Future/JointMax", "#0D8D00"),
    ("u_word", "Word/JointMax", "#AC7000"),
    ("u_pas", "Past/JointMax", "#A90008"),
    ("shared", "Shared/JointMax", "#1F77B4"),
]

SHARED_COMPONENTS = ["c_fp", "c_fw", "c_pw", "c_fpw"]
DEFAULT_LAGS_TO_PLOT = [-2000, -1000, -500, 0, 500, 1000, 2000]
DEFAULT_RATIO_THRESHOLD = 0.3
ALT_COMP_SPLIT_MS = 350
DEFAULT_OUTPUT_DIR = (
    "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/roi_figures/"
    "future_past/variance_partitioning/brainmaps"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot variance-partitioning ratio glassbrains. "
            "Ratios are component / max(total_r2 across lags) per electrode."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["prod", "comp", "both"],
        default="both",
        help="Which task to plot (default: prod).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=DEFAULT_LAGS_TO_PLOT,
        help="Requested lags in ms for the series plot.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Color scale minimum.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.7,
        help="Color scale maximum.",
    )
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help="Only plot electrodes with ratio strictly greater than this value.",
    )
    return parser.parse_args()


def _available_lags(df):
    lags = []
    for col in df.columns:
        if col.endswith("_total_r2"):
            lag_str = col.replace("_total_r2", "")
            try:
                lags.append(int(lag_str))
            except ValueError:
                continue
    if not lags:
        raise ValueError("No lag columns ending with '_total_r2' were found.")
    return sorted(set(lags))


def _nearest_available_lags(requested_lags, available_lags):
    mapped = []
    for lag in requested_lags:
        nearest = min(available_lags, key=lambda x: abs(x - lag))
        mapped.append(nearest)
    return mapped


def _compute_component_ratios(df, lags):
    base = df[["subject", "electrode", "roi"]].copy()
    base["subject"] = base["subject"].astype(str)

    denom_cols = [f"{lag}_total_r2" for lag in lags]
    joint_max = df[denom_cols].max(axis=1).replace(0, np.nan)

    ratios = {}
    for component_key, _, _ in COMPONENT_SPECS:
        comp_df = base.copy()

        for lag in lags:
            if component_key == "shared":
                numerator = sum(df[f"{lag}_{c}"] for c in SHARED_COMPONENTS)
            else:
                numerator = df[f"{lag}_{component_key}"]

            numerator = np.maximum(numerator, 0)
            comp_df[str(lag)] = numerator / joint_max

        ratios[component_key] = comp_df

    return ratios


def _effect_df(df_ratio, lag, ratio_thresh):
    lag_col = str(lag)
    out = df_ratio[["subject", "electrode", lag_col]].copy()
    out = out.rename(columns={lag_col: "effect"})
    out = out[out["effect"] > ratio_thresh].copy()
    return out


def _threshold_tag(ratio_thresh):
    return f"thr{int(round(ratio_thresh * 10)):02d}"


def _split_allowed_lags(component_key, lags_to_plot, split_ms=0, include_equal=True):
    if component_key == "u_fut":
        if include_equal:
            return [lag for lag in lags_to_plot if lag <= split_ms]
        return [lag for lag in lags_to_plot if lag < split_ms]

    if component_key == "u_pas":
        if include_equal:
            return [lag for lag in lags_to_plot if lag >= split_ms]
        return [lag for lag in lags_to_plot if lag > split_ms]

    return list(lags_to_plot)


def _build_component_colormaps():
    cmaps = {}
    for comp_key, _, hex_color in COMPONENT_SPECS:
        cmaps[comp_key] = mcolors.LinearSegmentedColormap.from_list(
            f"{comp_key}_cmap", ["white", hex_color], N=256
        )
    return cmaps


def _plot_ratio_series(ratios, lags_to_plot, output_path, title_prefix, vmin, vmax, ratio_thresh):
    colormaps = _build_component_colormaps()
    grayscale_cmap = mcolors.LinearSegmentedColormap.from_list(
        "context_gray_cmap", ["white", "grey"], N=256
    )

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

    for row_idx, (component_key, row_name, _) in enumerate(COMPONENT_SPECS):
        df_ratio = ratios[component_key]

        for col_idx, lag in enumerate(lags_to_plot):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if row_idx == 0:
                first_row_axes.append(ax)

            if (component_key == "u_fut" and lag > 0) or (component_key == "u_pas" and lag < 0):
                panel_cmap = grayscale_cmap
            else:
                panel_cmap = colormaps[component_key]

            pu.plot_effect_glassbrain(
                _effect_df(df_ratio, lag, ratio_thresh),
                effect_col="effect",
                cmap=panel_cmap,
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
            plt.cm.ScalarMappable(norm=norm, cmap=colormaps[component_key]),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label("Ratio", rotation=270, labelpad=15)

    for lag, ax in zip(lags_to_plot, first_row_axes):
        x_pos = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_pos, 0.95, f"{lag} ms", ha="center", va="center", fontsize=12, weight="bold")

    plt.suptitle(f"{title_prefix}: Variance-partition Ratio Brain Maps Across Lags", fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _pick_single_map_lags(ratios, lags_to_plot, split_ms=0, include_equal=True):
    picked = {}

    def best_lag(component_key, allowed_lags):
        df_ratio = ratios[component_key]
        lag_means = {
            lag: np.nanmean(pd.to_numeric(df_ratio[str(lag)], errors="coerce"))
            for lag in allowed_lags
        }
        return max(lag_means, key=lag_means.get)

    picked["u_word"] = best_lag("u_word", lags_to_plot)
    picked["shared"] = best_lag("shared", lags_to_plot)

    fut_lags = _split_allowed_lags("u_fut", lags_to_plot, split_ms=split_ms, include_equal=include_equal)
    if not fut_lags:
        raise ValueError("No allowed lags available for future single-map selection.")
    picked["u_fut"] = best_lag("u_fut", fut_lags)

    past_lags = _split_allowed_lags("u_pas", lags_to_plot, split_ms=split_ms, include_equal=include_equal)
    if not past_lags:
        raise ValueError("No allowed lags available for past single-map selection.")
    picked["u_pas"] = best_lag("u_pas", past_lags)

    return picked


def _plot_single_maps(ratios, picked_lags, output_path, title_prefix, vmin, vmax, ratio_thresh):
    colormaps = _build_component_colormaps()

    fig, axes = plt.subplots(1, len(COMPONENT_SPECS), figsize=(4 * len(COMPONENT_SPECS), 4))
    if len(COMPONENT_SPECS) == 1:
        axes = [axes]

    for ax, (component_key, label, _) in zip(axes, COMPONENT_SPECS):
        chosen_lag = picked_lags[component_key]
        panel_title = f"{label}\nLag {chosen_lag} ms"
        pu.plot_effect_glassbrain(
            _effect_df(ratios[component_key], chosen_lag, ratio_thresh),
            effect_col="effect",
            cmap=colormaps[component_key],
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            show=False,
            colorbar=True,
            title=None,
        )
        ax.set_title(
            panel_title,
            fontsize=12,
            color="black",
            pad=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, boxstyle="square,pad=0.2"),
        )

    plt.suptitle(f"{title_prefix}: Single-lag Component Ratio Brain Maps", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _effect_df_best_per_elec(df_ratio, allowed_lags, ratio_thresh):
    if not allowed_lags:
        raise ValueError("No allowed lags provided for per-electrode best-lag selection.")

    lag_cols = [str(lag) for lag in allowed_lags]
    vals = df_ratio[lag_cols].apply(pd.to_numeric, errors="coerce")

    out = df_ratio[["subject", "electrode"]].copy()
    out["effect"] = vals.max(axis=1)
    best_lag_cols = vals.idxmax(axis=1)
    out["best_lag"] = pd.to_numeric(best_lag_cols, errors="coerce")
    out = out[np.isfinite(out["effect"])].copy()
    out = out[out["effect"] > ratio_thresh].copy()
    return out


def _plot_single_maps_best_per_elec(
    ratios,
    lags_to_plot,
    output_path,
    title_prefix,
    vmin,
    vmax,
    ratio_thresh,
    split_ms=0,
    include_equal=True,
):
    colormaps = _build_component_colormaps()

    fig, axes = plt.subplots(1, len(COMPONENT_SPECS), figsize=(4 * len(COMPONENT_SPECS), 4))
    if len(COMPONENT_SPECS) == 1:
        axes = [axes]

    for ax, (component_key, label, _) in zip(axes, COMPONENT_SPECS):
        allowed_lags = _split_allowed_lags(
            component_key,
            lags_to_plot,
            split_ms=split_ms,
            include_equal=include_equal,
        )

        effect_df = _effect_df_best_per_elec(ratios[component_key], allowed_lags, ratio_thresh)
        panel_title = f"{label}\nBest lag per electrode"
        pu.plot_effect_glassbrain(
            effect_df,
            effect_col="effect",
            cmap=colormaps[component_key],
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            show=False,
            colorbar=True,
            title=None,
        )
        ax.set_title(
            panel_title,
            fontsize=12,
            color="black",
            pad=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, boxstyle="square,pad=0.2"),
        )

    plt.suptitle(f"{title_prefix}: Single-map Ratio Brain Maps (Best Lag Per Electrode)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _run_for_mode(
    df_mode,
    mode_name,
    requested_lags,
    output_dir,
    vmin,
    vmax,
    ratio_thresh,
    split_ms=0,
    include_equal=True,
    name_suffix="",
):
    os.makedirs(output_dir, exist_ok=True)

    available_lags = _available_lags(df_mode)
    lags_to_plot = _nearest_available_lags(requested_lags, available_lags)

    print(f"[{mode_name}] requested lags: {requested_lags}")
    print(f"[{mode_name}] using lags: {lags_to_plot}")
    print(f"[{mode_name}] best-lag search space: all {len(available_lags)} available lags")
    print(f"[{mode_name}] ratio threshold: > {ratio_thresh}")
    split_rule = "inclusive" if include_equal else "strict"
    print(f"[{mode_name}] split for future/past: {split_rule} at {split_ms} ms")

    ratios = _compute_component_ratios(df_mode, available_lags)
    thresh_tag = _threshold_tag(ratio_thresh)
    suffix = f"_{name_suffix}" if name_suffix else ""

    series_out_png = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_series_{thresh_tag}{suffix}.png")
    series_out_svg = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_series_{thresh_tag}{suffix}.svg")
    _plot_ratio_series(ratios, lags_to_plot, series_out_png, mode_name.capitalize(), vmin, vmax, ratio_thresh)
    _plot_ratio_series(ratios, lags_to_plot, series_out_svg, mode_name.capitalize(), vmin, vmax, ratio_thresh)

    picked_lags = _pick_single_map_lags(
        ratios,
        available_lags,
        split_ms=split_ms,
        include_equal=include_equal,
    )
    print(f"[{mode_name}] selected single-map lags: {picked_lags}")

    single_out_png = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_single_{thresh_tag}{suffix}.png")
    single_out_svg = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_single_{thresh_tag}{suffix}.svg")
    _plot_single_maps(ratios, picked_lags, single_out_png, mode_name.capitalize(), vmin, vmax, ratio_thresh)
    _plot_single_maps(ratios, picked_lags, single_out_svg, mode_name.capitalize(), vmin, vmax, ratio_thresh)

    single_best_out_png = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_single_best-per-elec_{thresh_tag}{suffix}.png")
    single_best_out_svg = os.path.join(output_dir, f"{mode_name}_variance_partition_ratio_single_best-per-elec_{thresh_tag}{suffix}.svg")
    _plot_single_maps_best_per_elec(
        ratios,
        available_lags,
        single_best_out_png,
        mode_name.capitalize(),
        vmin,
        vmax,
        ratio_thresh,
        split_ms=split_ms,
        include_equal=include_equal,
    )
    _plot_single_maps_best_per_elec(
        ratios,
        available_lags,
        single_best_out_svg,
        mode_name.capitalize(),
        vmin,
        vmax,
        ratio_thresh,
        split_ms=split_ms,
        include_equal=include_equal,
    )

    print(f"[{mode_name}] saved series: {series_out_png}")
    print(f"[{mode_name}] saved single: {single_out_png}")
    print(f"[{mode_name}] saved single (best per electrode): {single_best_out_png}")


def main():
    args = _parse_args()

    comp_df, prod_df = run_variance_partitioning_pipeline()

    if args.mode in ["prod", "both"]:
        _run_for_mode(
            df_mode=prod_df,
            mode_name="production",
            requested_lags=args.lags,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
            ratio_thresh=args.ratio_thresh,
        )
        # _run_for_mode(
        #     df_mode=prod_df,
        #     mode_name="production",
        #     requested_lags=args.lags,
        #     output_dir=args.output_dir,
        #     vmin=args.vmin,
        #     vmax=args.vmax,
        #     ratio_thresh=args.ratio_thresh,
        #     split_ms=0,
        #     include_equal=True,
        #     name_suffix="350ms",
        # )

    if args.mode in ["comp", "both"]:
        _run_for_mode(
            df_mode=comp_df,
            mode_name="comprehension",
            requested_lags=args.lags,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
            ratio_thresh=args.ratio_thresh,
        )
        _run_for_mode(
            df_mode=comp_df,
            mode_name="comprehension",
            requested_lags=args.lags,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
            ratio_thresh=args.ratio_thresh,
            split_ms=ALT_COMP_SPLIT_MS,
            include_equal=False,
            name_suffix="350ms",
        )


if __name__ == "__main__":
    main()
