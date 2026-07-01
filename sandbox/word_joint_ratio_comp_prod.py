#!/usr/bin/env python3
"""Plot comprehension vs production violin plots for word maximum effects.

This script reuses the same per-electrode banded ridge outputs as the plotting
code in `scripts/tfsplt_future_past_mistral_bands_brainmaps.py`.
It computes one value per electrode per mode in two versions:

    max_lag(word encoding) / max_lag(joint encoding)

and

    max_lag(word encoding)

and keeps only electrodes where the joint maximum exceeds 0.1 in both modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
 
# Add the 'scripts' directory to sys.path to import tfsplt_future_past_utils
sys.path.append(str(Path(__file__).parents[1] / "scripts"))

DEFAULT_RESULTS_DIR = Path(
    "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent
DEFAULT_SUBJECTS = ["625", "676", "7170", "798"]
MODEL_NAME = (
    "ij-tfs-{sid}-mistral-mistral_bandedRidge-lag30k-50-var-partition_"
    "pca300_all_load-splits_no-shift_cnxt8_deltasv2/ij-200ms-{sid}"
)

import tfsplt_future_past_utils as pu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--joint-thresh", type=float, default=0.1)
    return parser.parse_args()


def model_dir(results_dir: Path, sid: str) -> Path:
    return results_dir / MODEL_NAME.format(sid=sid)


def load_component(results_dir: Path, subjects: list[str], mode: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    file_re = re.compile(rf"^(?P<sid>\d+)_(?P<electrode>.+)_(?P<mode>comp|prod)_banded_{label}\.csv$")

    for sid in subjects:
        comp_dir = model_dir(results_dir, sid)
        if not comp_dir.exists():
            continue

        for file_path in sorted(comp_dir.glob(f"*_{mode}_banded_{label}.csv")):
            match = file_re.match(file_path.name)
            if match is None:
                continue

            mat = pd.read_csv(file_path, header=None)
            if mat.empty:
                continue

            values = mat.mean(axis=0).astype(float).to_numpy()
            row = {str(i): values[i] for i in range(len(values))}
            row["subject"] = match.group("sid")
            row["electrode"] = match.group("electrode")
            rows.append(row)

    if not rows:
        raise ValueError(f"No files loaded for mode={mode}, label={label}.")
    df = pd.DataFrame(rows)
    return pu.add_roi_label_to_results(df)


def compute_ratio_table(results_dir: Path, subjects: list[str], mode: str, joint_thresh: float) -> pd.DataFrame:
    joint_df = load_component(results_dir, subjects, mode, "joint")
    word_df = load_component(results_dir, subjects, mode, "word")

    key_cols = ["subject", "electrode"]
    lag_cols = sorted([c for c in joint_df.columns if c.isdigit()], key=lambda x: int(x))
    if not lag_cols:
        raise ValueError(f"No numeric lag columns found for mode={mode}.")

    merged = joint_df[key_cols + lag_cols].merge(word_df[key_cols + lag_cols], on=key_cols, suffixes=("_joint", "_word"))
    if merged.empty:
        raise ValueError(f"No overlapping electrodes found for mode={mode}.")

    joint_max = merged[[f"{c}_joint" for c in lag_cols]].max(axis=1).replace(0, np.nan)
    word_max = merged[[f"{c}_word" for c in lag_cols]].max(axis=1).clip(lower=0)

    out = merged[key_cols].copy()
    out = pu.add_roi_label_to_results(out)
    out["joint_max"] = joint_max
    out["word_max"] = word_max
    out["word_joint_ratio"] = word_max / joint_max
    out = out[np.isfinite(out["word_joint_ratio"]) & (out["joint_max"] > joint_thresh)].copy()
    out["mode"] = mode

    # Filter ROIs by electrode count (more than 10 electrodes)
    roi_counts = out.groupby('roi')['electrode'].nunique()
    valid_rois = roi_counts[roi_counts > 10].index
    out = out[out['roi'].isin(valid_rois)].copy()
    return out


def main() -> None:
    args = parse_args()
    comp_df = compute_ratio_table(args.results_dir, args.subjects, "comp", args.joint_thresh)
    prod_df = compute_ratio_table(args.results_dir, args.subjects, "prod", args.joint_thresh)

    # Ensure common ROIs across comp and prod for consistent plotting
    common_rois = set(comp_df['roi'].unique()) & set(prod_df['roi'].unique())
    comp_df = comp_df[comp_df['roi'].isin(common_rois)].copy()
    prod_df = prod_df[prod_df['roi'].isin(common_rois)].copy()

    comp_n = len(comp_df)
    prod_n = len(prod_df)

    # Define the base plot specifications
    base_plot_specs = [
        {
            "metric_col": "word_joint_ratio",
            "ylabel": "max(word encoding) / max(joint encoding)",
            "title_prefix": "Word ratio by electrode",
            "filename_stem": "word_joint_ratio_violin_comp_prod",
        },
        {
            "metric_col": "word_max",
            "ylabel": "max(word encoding)",
            "title_prefix": "Word max by electrode",
            "filename_stem": "word_max_raw_violin_comp_prod",
        },
        {
            "metric_col": "joint_max",
            "ylabel": "max(joint encoding)",
            "title_prefix": "Joint max by electrode",
            "filename_stem": "joint_max_violin_comp_prod",
        },
    ]

    # Loop through different plotting versions (colored points by ROI vs. uncolored)
    for color_points_by_roi in [True, False]:
        for spec in base_plot_specs:
            metric_col = spec["metric_col"]
            ylabel = spec["ylabel"]
            title = f"{spec['title_prefix']} (joint max > {args.joint_thresh:g})"
            base_filename = spec["filename_stem"]

            # Adjust filename based on whether points are colored by ROI
            if color_points_by_roi:
                output_filename = f"{base_filename}_colored_roi.png"
            else:
                output_filename = f"{base_filename}_uncolored_roi.png"
            output_path = args.output_dir / output_filename

            comp_label = f"Comprehension n={comp_n}"
            prod_label = f"Production n={prod_n}"

            plot_df = pd.concat([
                comp_df[[metric_col, "roi"]].assign(mode=comp_label),
                prod_df[[metric_col, "roi"]].assign(mode=prod_label)
            ]).rename(columns={metric_col: "value"})

            if color_points_by_roi:
                fig, ax = plt.subplots(figsize=(10, 7)) # Larger figure for colored points
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
            sns.violinplot(
                data=plot_df,
                x="mode",
                y="value",
                order=[comp_label, prod_label],
                inner="quart",
                linewidth=2,
                color='#ff7f0e',  # Matplotlib 'tab:orange'
                ax=ax,
                fill=False
            )

            stripplot_kwargs = {
                "data": plot_df,
                "x": "mode",
                "y": "value",
                "order": [comp_label, prod_label],
                "alpha": 0.6,
                "size": 3,
                "jitter": 0.05, # Decreased jitter
                "ax": ax,
            }
            if color_points_by_roi:
                stripplot_kwargs["hue"] = "roi"
            else:
                stripplot_kwargs["color"] = '#ff7f0e' # Match violin outline color
            
            sns.stripplot(**stripplot_kwargs)
            
            if color_points_by_roi:
                ax.legend(title="ROI", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False) # Legend outside
            else:
                # Remove legend if points are not colored by ROI
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
