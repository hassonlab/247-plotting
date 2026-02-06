"""
Directional difference calculation for brain imaging analyses.

This module provides a function to calculate directional performance differences
between control and rephrased conditions, preserving the sign of the difference.
"""

import pandas as pd
import numpy as np


def calc_max_diff_directional(
    df_control,
    df_reph,
    label3_val,
    time_window_cols,
    percentile=90,
    min_abs_diff=None,
):
    """
    Compute per-electrode max DIRECTIONAL (control - reph) within a lag window.
    
    Unlike calc_max_diff_percentile, this preserves the sign of the difference:
    - Positive values: control performs better than rephrase
    - Negative values: rephrase performs better than control
    
    The difference with the largest absolute magnitude is selected, but its sign is preserved.
    
    Parameters
    ----------
    df_control : pd.DataFrame
        DataFrame containing control condition results
    df_reph : pd.DataFrame
        DataFrame containing rephrase condition results
    label3_val : str
        Value to filter label3 column (e.g., 'sentence', 'sentence2')
    time_window_cols : list
        List of column names representing time lag windows
    percentile : int, default=90
        Percentile threshold for filtering electrodes
    min_abs_diff : float, optional
        Minimum absolute difference threshold. Electrodes with |max_diff| < min_abs_diff are excluded.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - subject: subject ID
        - subject_electrode: combined subject_electrode identifier
        - electrode: electrode number
        - label3: condition label
        - max_diff: directional difference (positive = control better, negative = reph better)
        - max_abs_diff: absolute magnitude of max_diff (for filtering)
        - max_diff_lag: lag at which max difference occurs
        - roi: region of interest label
        - percentile: percentile value used
        - percentile_threshold: threshold value for this percentile
        - above_percentile: boolean indicating if electrode exceeds threshold
        
    Example
    -------
    >>> # Calculate directional differences
    >>> diffs = calc_max_diff_directional(
    ...     original8_c,  
    ...     paraphrase8_c, 
    ...     "sentence",
    ...     time_window_future_comp,
    ...     percentile=0,
    ...     min_abs_diff=0.04
    ... )
    >>> 
    >>> # Plot with diverging colormap
    >>> import tfsplt_future_past_utils as pu
    >>> pu.plot_effect_glassbrain(
    ...     diffs,
    ...     cmap="RdBu_r",  # Red=control better, Blue=reph better
    ...     effect_col="max_diff",
    ...     vmin=-0.10,
    ...     vmax=0.10,
    ...     title="Directional Performance Difference"
    ... )
    """
    results = []

    df_control_filt = df_control[df_control["label3"] == label3_val].copy()
    df_reph_filt    = df_reph[df_reph["label3"] == label3_val].copy()

    # Prefer a stable unique key if present
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

    if min_abs_diff is not None:
        try:
            min_abs_diff = float(min_abs_diff)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_abs_diff must be a number or None, got: {min_abs_diff!r}") from exc

    for _, row in merged.iterrows():
        control_cols = [f"{col}_control" for col in time_window_cols if f"{col}_control" in row.index]
        reph_cols    = [f"{col}_reph" for col in time_window_cols if f"{col}_reph" in row.index]

        control_vals = row[control_cols].to_numpy()
        reph_vals    = row[reph_cols].to_numpy()

        # Directional differences: positive when control > reph
        diffs = control_vals - reph_vals
        
        # Find index of maximum absolute difference
        abs_diffs = np.abs(diffs)
        max_abs_idx = int(np.argmax(abs_diffs)) if len(abs_diffs) else 0
        
        # Keep the sign of the difference
        max_diff = float(diffs[max_abs_idx]) if len(diffs) else np.nan
        max_abs_diff = float(abs_diffs[max_abs_idx]) if len(abs_diffs) else np.nan
        max_diff_lag = time_window_cols[max_abs_idx] if len(time_window_cols) else np.nan

        # Threshold filter based on absolute magnitude
        if min_abs_diff is not None:
            if (not np.isfinite(max_abs_diff)) or (max_abs_diff < min_abs_diff):
                continue

        subject_electrode = row.get("subject_electrode", np.nan)
        electrode = row.get("electrode_control", row.get("electrode_reph", row.get("electrode", np.nan)))
        roi = row.get("roi_control", row.get("roi_reph", "unknown"))

        # Extract subject
        if pd.notna(subject_electrode):
            subject = str(subject_electrode).split("_", 1)[0]
        else:
            subject = row.get("subject", row.get("subject_control", row.get("subject_reph", np.nan)))

        results.append({
            "subject": subject,
            "subject_electrode": subject_electrode,
            "electrode": electrode,
            "label3": label3_val,
            "max_diff": max_diff,  # Directional: can be positive or negative
            "max_abs_diff": max_abs_diff,  # For reference/filtering
            "max_diff_lag": max_diff_lag,
            "roi": roi,
        })

    results_df = pd.DataFrame(results)

    # Ensure these exist for all percentiles (concat-friendly)
    results_df["percentile"] = int(percentile)
    results_df["percentile_threshold"] = np.nan
    results_df["above_percentile"] = False

    if len(results_df) > 0 and results_df["max_abs_diff"].notna().any():
        thr = np.percentile(results_df["max_abs_diff"].dropna().to_numpy(), percentile)
        results_df["percentile_threshold"] = thr
        results_df["above_percentile"] = results_df["max_abs_diff"] >= thr

    return results_df
