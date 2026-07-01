"""Utilities for plot_word_rephrases.ipynb.

Provides electrode selection based on joint_max in the original condition,
ROI result building, lineplots, and brainplot score computation for
comparing original / synonym / paronym / random conditions.
"""

import sys

# Make Gidon's shared helpers (tfsplt_future_past_helpers, tfsplt_future_past_utils)
# available when this module is imported from scripts/gidon/.
_GIDON_SCRIPTS = '/scratch/gpfs/HASSON/gidon/247-plotting/scripts'
if _GIDON_SCRIPTS not in sys.path:
    sys.path.insert(0, _GIDON_SCRIPTS)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from tfsplt_future_past_helpers import pair_metrics


# ---------------------------------------------------------------------------
# Electrode selection
# ---------------------------------------------------------------------------

def _lag_columns(df, lag_min, lag_max):
    """Return column names whose numeric value falls in [lag_min, lag_max]."""
    cols = []
    for c in df.columns:
        try:
            v = int(c)
            if lag_min <= v <= lag_max:
                cols.append(c)
        except (ValueError, TypeError):
            pass
    return cols


def select_electrodes_by_joint_max(original_df, threshold=0.1,
                                   lag_min=-2000, lag_max=2000):
    """Return the set of electrode IDs where max(joint) >= threshold.

    Looks at the 'joint' label3 rows of *original_df* and returns the
    electrode identifiers (as strings) whose maximum value over
    [lag_min, lag_max] meets the threshold.
    """
    joint_df = original_df[original_df['label3'] == 'joint']
    lag_cols = _lag_columns(joint_df, lag_min, lag_max)
    if not lag_cols:
        raise ValueError(
            f"No lag columns found in range [{lag_min}, {lag_max}]. "
            "Make sure align_lag_columns() has been called first."
        )
    joint_max = joint_df[lag_cols].max(axis=1)
    elec_col = 'subject_electrode' if 'subject_electrode' in joint_df.columns else 'electrode'
    return set(joint_df.loc[joint_max >= threshold, elec_col].astype(str))


def intersect_conditions(condition_dfs, electrode_set):
    """Filter each DataFrame to *electrode_set*, then intersect across all conditions.

    Parameters
    ----------
    condition_dfs : dict[str, pd.DataFrame]
        Mapping condition_name → DataFrame.
    electrode_set : set[str]
        Electrode IDs pre-selected (e.g., from joint_max threshold).

    Returns
    -------
    filtered : dict[str, pd.DataFrame]
    common_electrodes : set[str]
    """
    elec_col = None
    for df in condition_dfs.values():
        if 'subject_electrode' in df.columns:
            elec_col = 'subject_electrode'
            break
        if 'electrode' in df.columns:
            elec_col = 'electrode'
    if elec_col is None:
        raise KeyError("No 'electrode' or 'subject_electrode' column found.")

    # Filter each df to electrode_set
    filtered = {
        name: df[df[elec_col].astype(str).isin(electrode_set)].copy()
        for name, df in condition_dfs.items()
    }

    # Intersect across all conditions
    common = None
    for df in filtered.values():
        elecs = set(df[elec_col].astype(str))
        common = elecs if common is None else common & elecs
    common = common or set()

    filtered = {
        name: df[df[elec_col].astype(str).isin(common)].copy()
        for name, df in filtered.items()
    }
    return filtered, common


# ---------------------------------------------------------------------------
# ROI result building
# ---------------------------------------------------------------------------

def build_roi_results(filtered_dfs_dict, condition_names, rois,
                      results=('joint', 'word', 'sentence', 'sentence2')):
    """Build a roi_results dict keyed by (condition_idx, label3, roi).

    Parameters
    ----------
    filtered_dfs_dict : dict[str, pd.DataFrame]
        Already filtered DataFrames keyed by condition name.
    condition_names : list[str]
        Ordered list of condition names (index → key into filtered_dfs_dict).
    rois : list[str]
        ROI names to iterate over. 'All' means no ROI filter.

    Returns
    -------
    dict[(int, str, str), pd.DataFrame]
    """
    roi_results = {}
    for i, name in enumerate(condition_names):
        df = filtered_dfs_dict[name]
        roi_col = 'roi' if 'roi' in df.columns else None
        for result in results:
            result_mask = df['label3'] == result
            for roi in rois:
                if roi == 'All' or roi_col is None:
                    sub = df[result_mask].copy()
                else:
                    sub = df[result_mask & (df[roi_col] == roi)].copy()
                if len(sub) > 0:
                    roi_results[(i, result, roi)] = sub
    return roi_results


# ---------------------------------------------------------------------------
# Lag-column alignment (needed before brainplot score computation)
# ---------------------------------------------------------------------------

def align_lag_columns(dfs, lag_cols, n_meta_cols=5):
    """Rename the leading lag columns of each DataFrame to *lag_cols* values.

    Modifies DataFrames in-place.  Assumes the last *n_meta_cols* columns are
    metadata and everything before them are lag columns.
    """
    for df in dfs:
        n_lag = len(df.columns) - n_meta_cols
        new_cols = list(lag_cols[:n_lag]) + list(df.columns[-n_meta_cols:])
        df.columns = new_cols


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------

def pink_grey_yellow_cmap(name='rephrase_cmap'):
    """Pink → grey → yellow diverging colormap (matches Gidon's convention)."""
    return LinearSegmentedColormap.from_list(
        name, ['#c2005c', '#9e9e9e', '#ffd700'], N=256
    )


def paronym_grey_synonym_cmap(name='word_rephrase_cmap'):
    """Orange → grey → blue colormap matching lineplot condition colors.

    Orange (#FF5722) = paronym, Blue (#2196F3) = synonym.
    Positive score (synonym > paronym) → blue end.
    """
    return LinearSegmentedColormap.from_list(
        name, ['#FF5722', '#9E9E9E', '#2196F3'], N=256
    )


# ---------------------------------------------------------------------------
# Brainplot rendering helpers
# ---------------------------------------------------------------------------

_COORDS_DIR = (
    "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting"
    "/data/plotting/brainplot/"
)
_COORD_SUBJECTS = ["625", "676", "717", "798"]


def filter_to_known_coords(df, coords_dir=_COORDS_DIR, subjects=_COORD_SUBJECTS):
    """Keep only rows whose (subject, electrode) appears in the MNI coordinate file.

    Mirrors the merge done inside ``plot_effect_glassbrain`` so that nilearn's
    ``plot_markers`` never receives NaN coordinate rows, which cause the
    "None is not a valid value for color" crash.
    """
    from tfsplt_brainmap import read_coor
    df_coor = read_coor(coords_dir, list(subjects))
    df_coor.loc[df_coor['subject'] == '717', 'subject'] = '7170'
    df_coor = (
        df_coor.rename(columns={"name": "electrode"})[['subject', 'electrode']]
        .astype(str)
        .drop_duplicates()
    )
    out = df.copy()
    out['subject']   = out['subject'].astype(str)
    out['electrode'] = out['electrode'].astype(str)
    return out.merge(df_coor, on=['subject', 'electrode'], how='inner')


def render_raw_diff_brainplots(
    electrode_scores,
    plot_order,
    cmaps,
    colorbar_labels,
    use_size_encoding=True,
):
    """Render brainplots colored by raw synonym − paronym difference (Pearson r).

    Uses ``reph_at_max − control_at_max`` from the compute_brainplot_scores
    output as the color effect — the mean synonym encoding minus mean paronym
    encoding at the top-N lags where their difference is largest.

    Color scale is symmetric and auto-scaled to the data (no fixed ±75).
    NaN scores are dropped before plotting.

    Parameters
    ----------
    electrode_scores : pd.DataFrame
        Output of compute_brainplot_scores (concatenated across panels).
    plot_order : list of (mode, label3, desc)
    cmaps : dict[str, Colormap]
        Keyed by label3.
    colorbar_labels : dict[str, str]
    use_size_encoding : bool
        If True, use the ``node_size`` column for marker size.
    """
    import tfsplt_future_past_utils as _pu

    df = electrode_scores.copy()
    df['raw_diff'] = df['reph_at_max'] - df['control_at_max']

    # Symmetric auto-scaled colorbar across all panels
    finite_vals = df['raw_diff'].dropna().to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        print("No finite raw_diff values to plot.")
        return
    vmax = float(np.nanmax(np.abs(finite_vals)))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    print(f"raw_diff brainplots | vmin={vmin:.4f}  vmax={vmax:.4f}")

    for mode, label3, desc in plot_order:
        panel = df[(df['mode'] == mode) & (df['label3'] == label3)].copy()
        panel = panel[panel['raw_diff'].notna() & np.isfinite(panel['raw_diff'])]

        # Keep only electrodes with known MNI coordinates so nilearn's
        # plot_markers never receives NaN coordinate rows.
        panel = filter_to_known_coords(panel)

        if len(panel) == 0:
            print(f"  {desc}: 0 electrodes after coordinate filter — skipping")
            continue

        print(f"  {desc}: {len(panel)} electrodes")

        node_size_col = 'node_size' if use_size_encoding else None
        ax = _pu.plot_effect_glassbrain(
            panel,
            cmap=cmaps[label3],
            effect_col='raw_diff',
            vmin=vmin,
            vmax=vmax,
            sign_color_mode='off',
            title=f"{desc} (raw diff, r units)",
            node_size_col=node_size_col,
        )

        if ax is not None and ax.figure is not None and len(ax.figure.axes) > 1:
            cbar_ax = ax.figure.axes[-1]
            cbar_ax.set_ylabel(colorbar_labels[label3], rotation=270, labelpad=18)
            cbar_ax.yaxis.set_label_position('right')


def render_rephrase_brainplot_panels(
    electrode_scores,
    plot_order,
    cmaps,
    colorbar_labels,
    score_mode,
    size_pair_labels=('original', 'comparison'),
    use_size_encoding=True,
    top_k=None,
    panel_thresholds=None,
):
    """Same as tfsplt_future_past_helpers.render_brainplot_panels, but the
    on-plot size-legend text is configurable via *size_pair_labels* instead
    of being hardcoded to ("original", "double rephrase").

    Use this whenever SIZE_PAIR isn't literally (original, double-rephrase)
    -- e.g. (original, random) or (original, shuffled) -- so the overlay
    text on the brainplot matches what size actually encodes.
    """
    import tfsplt_future_past_utils as _pu
    from tfsplt_future_past_helpers import add_size_legend

    if len(electrode_scores) == 0:
        print("No rows after thresholding.")
        return

    effect_col = "score"
    vals = electrode_scores[effect_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax_auto = float(np.nanmax(vals)) if vals.size else 1.0
    vmax_auto = max(vmax_auto, 1e-6)

    if score_mode == "percent_change_at_max_diff":
        vmin, vmax = -75, 75
    elif score_mode == "max_abs_diff":
        vmin, vmax = 0.0, vmax_auto
    else:
        raise ValueError("Unsupported score_mode")

    print(f"score_mode={score_mode} | n_total={len(electrode_scores)}")
    for mode, label3, desc in plot_order:
        panel = electrode_scores[
            (electrode_scores["mode"] == mode) & (electrode_scores["label3"] == label3)
        ].copy()
        if top_k is not None and len(panel) > top_k:
            panel = panel.nlargest(top_k, "score_abs")

        thresholds = {}
        if panel_thresholds:
            thresholds = panel_thresholds.get((mode, label3), {}) or {}

        if "value_thresholds" in thresholds:
            for cond, minv in thresholds["value_thresholds"].items():
                col = f"max_{cond}"
                if col in panel.columns:
                    panel = panel[panel[col].fillna(-np.inf) >= float(minv)]
                else:
                    panel = panel.iloc[0:0]

        if "min_max_diff" in thresholds:
            if "max_diff" in panel.columns:
                panel = panel[panel["max_diff"].fillna(-np.inf) >= float(thresholds["min_max_diff"])]
            else:
                panel = panel.iloc[0:0]

        if "band_original_min" in thresholds:
            if "max_original" in panel.columns:
                panel = panel[panel["max_original"].fillna(-np.inf) >= float(thresholds["band_original_min"])]
            else:
                panel = panel.iloc[0:0]

        if "min_score_abs" in thresholds:
            if "score" in panel.columns:
                panel = panel[np.abs(panel["score"].fillna(-np.inf)) >= float(thresholds["min_score_abs"])]
            else:
                panel = panel.iloc[0:0]

        if len(panel) == 0:
            print(f"    → {desc}: 0 electrodes (all filtered out)")
            continue

        elec_col = "subject_electrode" if "subject_electrode" in panel.columns else "electrode"
        elec_list = sorted(panel[elec_col].dropna().astype(str).tolist())
        print(f"    → {desc}: {len(panel)} electrodes plotted: {elec_list}")

        ax = _pu.plot_effect_glassbrain(
            panel,
            cmap=cmaps[label3],
            effect_col=effect_col,
            vmin=vmin,
            vmax=vmax,
            sign_color_mode="off",
            title=desc,
        )

        if ax is not None and ax.figure is not None and len(ax.figure.axes) > 1:
            cbar_ax = ax.figure.axes[-1]
            cbar_ax.set_ylabel(colorbar_labels[label3], rotation=270, labelpad=18)
            cbar_ax.yaxis.set_label_position("right")

            if use_size_encoding:
                add_size_legend(ax, size_min=1, size_max=99, size_pair_labels=size_pair_labels)


# ---------------------------------------------------------------------------
# Lineplots
# ---------------------------------------------------------------------------

def plot_rephrase_lineplots(args, roi_results, roi, titles, mode='comp',
                            ymax=0.18, plot_bands=True, save=False, bands=None):
    """Overlay all conditions on a single ROI lineplot.

    When *bands* is given, exactly those label3 bands are shown (one subplot
    each), in order. Otherwise: if *plot_bands* is True, only the 'word' band
    is shown (one subplot); if False, all bands in args.lines are shown.

    Parameters
    ----------
    args : argparse.Namespace
        Must have: .lines, .colors, .lags, .legends, .res_dir
    roi_results : dict
        Keyed by (condition_idx, label3, roi).
    roi : str
    titles : list[str]
        One title per condition (in condition_names order).
    mode : str
        'comp' or 'prod'.
    bands : list[str], optional
        Explicit list of label3 bands to plot as separate subplots, e.g.
        ['sentence', 'sentence2']. Overrides plot_bands when given.
    """
    dataset_indices = sorted({k[0] for k in roi_results})
    if not dataset_indices:
        return

    if bands is not None:
        lines_to_plot = list(bands)
    else:
        lines_to_plot = ['word'] if plot_bands else list(args.lines)
    n_lines = len(lines_to_plot)

    if n_lines == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_lines, figsize=(20, 5))

    dataset_colors = [args.colors[i] for i in dataset_indices]
    lags_all = args.lags['lags_all']
    lags_plt = args.lags['lags_plt']
    lags_sel = [li for li, lag in enumerate(lags_all) if lag in lags_plt]

    for subplot_idx, line in enumerate(lines_to_plot):
        ax = axes[subplot_idx]
        line_idx = args.lines.index(line) if line in args.lines else 0

        plotted_any = False
        for i, ds_idx in enumerate(dataset_indices):
            key = (ds_idx, line, roi)
            if key not in roi_results or roi_results[key].empty:
                continue
            df_line = roi_results[key]
            # iloc: lag columns are the leading columns (metadata are last 5)
            n_meta = 5
            n_lag = len(df_line.columns) - n_meta
            sel = [s for s in lags_sel if s < n_lag]
            vals = df_line.iloc[:, sel].mean(axis=0).values
            errs = df_line.iloc[:, sel].sem(axis=0).values
            x = lags_plt[:len(vals)]
            n_elecs = len(df_line)
            label = f"{titles[ds_idx]} (n={n_elecs})"
            ax.plot(x, vals, color=dataset_colors[i], label=label, lw=2.5)
            ax.fill_between(x, vals - errs, vals + errs,
                            alpha=0.2, color=dataset_colors[i])
            plotted_any = True

        if not plotted_any:
            ax.set_visible(False)
            continue

        if ymax is not None:
            ax.set_ylim(top=ymax, bottom=-0.025)
        ymin_val, ymax_val = ax.get_ylim()

        if mode == 'comp':
            ax.add_patch(patches.Rectangle(
                (50, ymin_val), 450, ymax_val - ymin_val,
                color='yellowgreen', alpha=0.3))
        else:
            ax.add_patch(patches.Rectangle(
                (-500, ymin_val), 450, ymax_val - ymin_val,
                color='indianred', alpha=0.3))

        ax.axhline(0, ls='dashed', alpha=0.3, c='k')
        ax.axvline(0, ls='dashed', alpha=0.3, c='k')
        ax.set_xticks(args.lags['lag_ticks'])
        ax.set_xticklabels(args.lags['lag_tick_labels'])
        if subplot_idx == 0:
            ax.set_ylabel("Pearson's r")
            ax.set_xlabel("time wrt. word-onset (s)")
        ax.tick_params(labelsize=10)
        ax.legend(loc='best', frameon=False, fontsize=9)
        if n_lines > 1:
            ax.set_title(args.legends[line_idx], fontsize=12)

    mode_full = 'Comprehension' if mode == 'comp' else 'Production'
    fig.suptitle(f"{roi} — {mode_full}", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(f"{args.res_dir}/{roi}_rephrase_{mode}.jpeg",
                    bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Brainplot score computation
# (adapted from Gidon's tfsplt_future_past_reph_cnxt_0008_updated.ipynb)
# ---------------------------------------------------------------------------

TOP_N_LAGS = 10


def compute_brainplot_scores(
    condition_dfs,
    label3,
    time_window,
    color_pair=('synonym', 'paronym'),
    *,
    value_thresholds=None,
    diff_thresholds=None,
    score_mode='max_abs_diff',
    size_pair=None,
    size_time_window=None,
    size_mode='max_abs_diff',
    size_min=10,
    size_max=90,
    band_thresholds=None,
    threshold_time_window=None,
    denominator_pair=('original', 'random'),
    threshold_label3=None,
    top_n_lags=None,
    own_band_value_thresholds=None,
    min_diff_any=None,
):
    """Compute per-electrode brainplot color and size scores.

    Parameters
    ----------
    condition_dfs : dict[str, pd.DataFrame]
        Keys must include color_pair, size_pair, denominator_pair, and
        value_thresholds keys.
    label3 : str
        Which label3 band to score ('joint', 'word', 'sentence', 'sentence2').
    time_window : list[int]
        Lag column values (after align_lag_columns) to use for scoring.
    color_pair : tuple[str, str]
        (condition_a, condition_b); positive score = a > b.
    value_thresholds : dict[str, float], optional
        {condition_name: min_max_value}.  Electrodes failing any threshold are
        excluded.
    diff_thresholds : list[tuple[str, str, float]], optional
        Each entry (cond_a, cond_b, min_diff); electrodes where
        |max(a) − max(b)| < min_diff are excluded.
    score_mode : str
        'max_abs_diff' or 'percent_change_at_max_diff'.
    size_pair : tuple[str, str], optional
        Pair used for node size encoding.  Defaults to color_pair.
    size_time_window : list[int], optional
        Defaults to time_window.
    size_mode : str
        'max_abs_diff' or 'percent_change_at_max_diff'.
    band_thresholds : dict[str, dict], optional
        {label3_val: {'value': float, 'cols': list[int]}}.
        Checked against the 'original' condition.
    threshold_time_window : dict[str, list[int]] or list[int], optional
        Per-condition time windows for value_thresholds.
    denominator_pair : tuple[str, str]
        Pair used as denominator for percent_change scoring.
    threshold_label3 : str, optional
        If given, value_thresholds are checked on a different label3 row.
    top_n_lags : int, optional
        Override module-level TOP_N_LAGS.
    own_band_value_thresholds : dict[str, float], optional
        {condition_name: min_peak}.  Like value_thresholds, but checked against
        the electrode's peak on *this* label3 band directly (no threshold_label3
        lookup) -- e.g. {"original": 0.025} to require a minimal word-band peak.
    min_diff_any : list[tuple[str, str, float]], optional
        Each entry (cond_a, cond_b, min_diff).  Electrode is kept if ANY entry's
        |max(cond_a) - max(cond_b)| >= min_diff (OR semantics, unlike
        diff_thresholds which requires ALL entries to pass).  Use this to keep
        electrodes where rephrasing measurably changed the encoding for at
        least one variant, while excluding electrodes where neither synonym
        nor paronym differs from original (no effect of rephrasing at all).

    Returns
    -------
    pd.DataFrame with columns: subject, subject_electrode, electrode, roi,
        label3, score, score_abs, score_mode, max_diff, percent_change,
        max_diff_lag, control_at_max, reph_at_max, size_score, size_score_abs,
        size_mode, size_max_diff, size_percent_change, node_size,
        max_<condition> for every condition in *needed*.
    """
    value_thresholds = value_thresholds or {}
    diff_thresholds = diff_thresholds or []
    band_thresholds = band_thresholds or {}
    own_band_value_thresholds = own_band_value_thresholds or {}
    min_diff_any = min_diff_any or []
    size_pair = color_pair if size_pair is None else size_pair
    size_time_window = time_window if size_time_window is None else size_time_window
    threshold_time_window = (time_window if threshold_time_window is None
                              else threshold_time_window)
    _top_n = int(top_n_lags) if top_n_lags is not None else int(TOP_N_LAGS)

    valid_modes = {'max_abs_diff', 'percent_change_at_max_diff'}
    if score_mode not in valid_modes or size_mode not in valid_modes:
        raise ValueError(f"score_mode and size_mode must be in {valid_modes}")
    if size_max < size_min:
        raise ValueError("size_max must be >= size_min")

    def _safe_nanmax(x):
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        return float(x.max()) if x.size else np.nan

    def _thresh_cols_for(cond):
        if isinstance(threshold_time_window, dict):
            return threshold_time_window.get(cond, time_window)
        return threshold_time_window

    def _mode_score(mode, pm, _a, _b):
        if mode == 'max_abs_diff':
            return pm['max_diff'], pm['max_diff']
        score_value = pm['pct']
        return score_value, (np.abs(score_value) if np.isfinite(score_value)
                              else np.nan)

    needed = {
        color_pair[0], color_pair[1],
        size_pair[0], size_pair[1],
        denominator_pair[0], denominator_pair[1],
        *value_thresholds.keys(),
        *own_band_value_thresholds.keys(),
    }
    for thr in diff_thresholds:
        if len(thr) != 3:
            raise ValueError("diff_threshold entries must be (cond_a, cond_b, min_diff)")
        needed.update([thr[0], thr[1]])
    for thr in min_diff_any:
        if len(thr) != 3:
            raise ValueError("min_diff_any entries must be (cond_a, cond_b, min_diff)")
        needed.update([thr[0], thr[1]])
    if label3 in band_thresholds:
        needed.add('original')

    missing = [c for c in needed if c not in condition_dfs]
    if missing:
        raise KeyError(f"Missing conditions in condition_dfs: {missing}")

    # Infer merge key
    def _infer_merge_keys(dfs_):
        cols = [set(df.columns) for df in dfs_]
        if all('subject_electrode' in c for c in cols):
            return ['subject_electrode']
        if all({'subject', 'electrode'}.issubset(c) for c in cols):
            return ['subject', 'electrode']
        if all('electrode' in c for c in cols):
            return ['electrode']
        raise KeyError("No common merge key found across conditions.")

    merge_keys = _infer_merge_keys([condition_dfs[c] for c in needed])

    # Collect all lag columns needed
    all_cols_set = set(time_window) | set(size_time_window)
    for band_info in band_thresholds.values():
        if isinstance(band_info, dict) and 'cols' in band_info:
            all_cols_set |= set(band_info['cols'])
    all_needed_cols = sorted(all_cols_set)

    # Optional: threshold lookup per electrode when threshold_label3 is set
    thresh_label3_lookup = {}
    if threshold_label3 is not None and value_thresholds:
        for cond in value_thresholds:
            df_t = condition_dfs[cond]
            df_t = df_t[df_t['label3'] == threshold_label3]
            tcols = [c for c in _thresh_cols_for(cond) if c in df_t.columns]
            lookup = {}
            for _, row_t in df_t.iterrows():
                elec_key = tuple(str(row_t[k]) for k in merge_keys)
                lookup[elec_key] = _safe_nanmax([row_t[c] for c in tcols])
            thresh_label3_lookup[cond] = lookup

    # Merge all conditions on shared electrode key
    merged = None
    for idx, cond in enumerate(sorted(needed)):
        df = condition_dfs[cond]
        df = df[df['label3'] == label3].copy()
        missing_cols = [c for c in all_needed_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Condition '{cond}' missing lag columns: "
                f"{missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
            )
        keep_meta = list(merge_keys)
        if idx == 0:
            for meta_col in ['roi', 'subject', 'subject_electrode', 'electrode']:
                if meta_col in df.columns and meta_col not in keep_meta:
                    keep_meta.append(meta_col)
        part = df[keep_meta + all_needed_cols].rename(
            columns={c: f"{c}__{cond}" for c in all_needed_cols}
        )
        merged = part if merged is None else merged.merge(part, on=merge_keys, how='inner')

    denom_a, denom_b = denominator_pair
    rows = []
    for _, row in merged.iterrows():
        def vals(cond, cols):
            return np.asarray([row[f"{c}__{cond}"] for c in cols], dtype=float)

        cond_max = {c: _safe_nanmax(vals(c, time_window)) for c in needed}

        if thresh_label3_lookup:
            threshold_max = {}
            for cond in value_thresholds:
                elec_key = tuple(str(row[k]) for k in merge_keys)
                threshold_max[cond] = thresh_label3_lookup[cond].get(elec_key, np.nan)
        else:
            threshold_max = {
                c: _safe_nanmax(vals(c, _thresh_cols_for(c)))
                for c in value_thresholds
            }

        if not all(threshold_max.get(c, np.nan) >= v
                   for c, v in value_thresholds.items()):
            continue

        if own_band_value_thresholds and not all(
            cond_max.get(c, np.nan) >= v for c, v in own_band_value_thresholds.items()
        ):
            continue

        ok_diff = True
        for ca, cb, min_diff in diff_thresholds:
            va = _safe_nanmax(vals(ca, time_window))
            vb = _safe_nanmax(vals(cb, time_window))
            if not np.isfinite(va) or not np.isfinite(vb) or abs(va - vb) < min_diff:
                ok_diff = False
                break
        if not ok_diff:
            continue

        if min_diff_any:
            diff_any_ok = False
            for ca, cb, min_diff in min_diff_any:
                va = cond_max.get(ca, np.nan)
                vb = cond_max.get(cb, np.nan)
                if np.isfinite(va) and np.isfinite(vb) and abs(va - vb) >= min_diff:
                    diff_any_ok = True
                    break
            if not diff_any_ok:
                continue

        if label3 in band_thresholds:
            bi = band_thresholds[label3]
            band_max = _safe_nanmax(vals('original', bi.get('cols', time_window)))
            if not np.isfinite(band_max) or band_max < bi.get('value', 0):
                continue

        ca, cb = color_pair
        pm_c = pair_metrics(
            vals(ca, time_window), vals(cb, time_window),
            top_n=_top_n,
            size_a=vals(denom_a, time_window),
            size_b=vals(denom_b, time_window),
        )
        score, score_abs = _mode_score(score_mode, pm_c,
                                        cond_max.get(ca), cond_max.get(cb))

        # Skip electrodes with an undefined score (e.g. denominator ≈ 0 in
        # percent_change_at_max_diff).  Passing NaN as a color crashes matplotlib.
        if not np.isfinite(score):
            continue

        sa, sb = size_pair
        pm_s = pair_metrics(vals(sa, size_time_window), vals(sb, size_time_window),
                             top_n=_top_n)
        size_score, size_score_abs = _mode_score(
            size_mode, pm_s,
            _safe_nanmax(vals(sa, size_time_window)),
            _safe_nanmax(vals(sb, size_time_window)),
        )

        # Build subject_electrode.  If the data only carries subject + electrode
        # separately, construct the combined key so render_brainplot_panels can
        # use it; otherwise it picks the all-NaN column and reports "[] electrodes".
        se = row.get('subject_electrode', np.nan)
        subject_val  = row.get('subject', np.nan)
        electrode_val = row.get('electrode', np.nan)
        if pd.isna(se) and pd.notna(subject_val) and pd.notna(electrode_val):
            se = f"{subject_val}_{electrode_val}"
        subject = (str(se).split('_', 1)[0] if pd.notna(se) else subject_val)

        out = {
            'subject':           subject,
            'subject_electrode': se,
            'electrode':         electrode_val,
            'roi':              row.get('roi', 'unknown'),
            'label3':           label3,
            'score':            score,
            'score_abs':        score_abs,
            'score_mode':       score_mode,
            'max_diff':         pm_c['max_diff'],
            'percent_change':   pm_c['pct'],
            'max_diff_lag':     (time_window[pm_c['idx']]
                                  if pm_c['idx'] < len(time_window) else np.nan),
            'control_at_max':   pm_c['b_at'],
            'reph_at_max':      pm_c['a_at'],
            'size_score':       size_score,
            'size_score_abs':   size_score_abs,
            'size_mode':        size_mode,
            'size_max_diff':    pm_s['max_diff'],
            'size_percent_change': pm_s['pct'],
        }
        for cond in needed:
            out[f"max_{cond}"] = cond_max.get(cond, np.nan)
        rows.append(out)

    df_out = pd.DataFrame(rows)
    df_out['node_size'] = float(size_min)

    if len(df_out) and df_out['size_score_abs'].notna().any():
        vals_arr = df_out['size_score_abs'].to_numpy(dtype=float)
        vmin, vmax = float(np.nanmin(vals_arr)), float(np.nanmax(vals_arr))
        if vmax > vmin:
            scaled = size_min + ((vals_arr - vmin) / (vmax - vmin)) * (size_max - size_min)
        else:
            scaled = np.full(len(vals_arr),
                              float(size_max if np.isfinite(vmax) and vmax > 0
                                    else size_min))
        df_out['node_size'] = np.clip(scaled, float(size_min), float(size_max))

    return df_out
