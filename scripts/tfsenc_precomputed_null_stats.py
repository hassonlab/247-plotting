import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", nargs="+", type=int, required=True)
    parser.add_argument("--keys", nargs="+", default=["comp", "prod"])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--null-dir", required=True)
    parser.add_argument("--result-tail", default="_banded_joint.csv")
    parser.add_argument("--lags-plot", nargs="+", type=int, required=True)
    parser.add_argument("--q-thresh", type=float, default=0.05)
    parser.add_argument("--outfile", required=True)
    return parser.parse_args()


def resolve_sid_path(path_or_pattern, sid):
    if "{sid}" in path_or_pattern:
        return path_or_pattern.format(sid=sid)
    return path_or_pattern


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def load_observed_curve(fname):
    df = pd.read_csv(fname, header=None)
    if len(df) > 1:
        df = df.iloc[[-1]]
    return df.iloc[0].to_numpy(dtype=float)


def main():
    args = parse_args()
    records = []
    missing_null = 0
    missing_obs = 0

    lags_plot = np.asarray(args.lags_plot, dtype=int)

    for sid in args.sid:
        sid_results_root = resolve_sid_path(args.results_dir, sid)
        sid_null_root = resolve_sid_path(args.null_dir, sid)
        sid_results_dir = os.path.join(sid_results_root, f"ij-200ms-{sid}")
        sid_null_dir = os.path.join(sid_null_root, f"ij-200ms-{sid}", "null_distributions")

        for key in args.keys:
            pattern = os.path.join(sid_results_dir, f"{sid}_*_{key}{args.result_tail}")
            obs_files = sorted(glob.glob(pattern))
            if len(obs_files) == 0:
                missing_obs += 1
                print(f"No observed files found: {pattern}")
                continue

            for obs_file in obs_files:
                base = os.path.basename(obs_file)
                suffix = f"_{key}{args.result_tail}"
                if not base.endswith(suffix):
                    continue

                stem = base[: -len(suffix)]
                sid_prefix = f"{sid}_"
                if not stem.startswith(sid_prefix):
                    continue

                elec_name = stem[len(sid_prefix) :]
                electrode = f"{sid}_{elec_name}"
                null_file = os.path.join(
                    sid_null_dir,
                    f"{sid}_{elec_name}_{key}_null_perf.h5",
                )

                if not os.path.exists(null_file):
                    missing_null += 1
                    continue

                observed = load_observed_curve(obs_file)
                if len(observed) != len(lags_plot):
                    raise ValueError(
                        f"Observed curve length mismatch for {obs_file}: "
                        f"len(curve)={len(observed)} vs len(lags_plot)={len(lags_plot)}"
                    )

                with h5py.File(null_file, "r") as h5f:
                    null_corrs = h5f["null_corrs"][:]
                    lag_indices = h5f["lag_indices"][:].astype(int)

                lag_to_null_col = {lag_idx: j for j, lag_idx in enumerate(lag_indices)}
                for lag_idx in lag_indices:
                    if lag_idx < 0 or lag_idx >= len(observed):
                        continue
                    null_col = lag_to_null_col[lag_idx]
                    obs_val = float(observed[lag_idx])
                    null_vals = null_corrs[:, null_col]
                    pval = (1.0 + np.sum(null_vals >= obs_val)) / (len(null_vals) + 1.0)
                    records.append(
                        {
                            "sid": sid,
                            "electrode": electrode,
                            "key": key,
                            "lag_index": int(lag_idx),
                            "lag_ms": int(lags_plot[lag_idx]),
                            "observed_r": obs_val,
                            "p_value": float(pval),
                        }
                    )

    if len(records) == 0:
        raise RuntimeError("No test records were generated.")

    out_df = pd.DataFrame.from_records(records)
    out_df["q_value"] = np.nan
    for key, key_idx in out_df.groupby("key").groups.items():
        qvals = bh_fdr(out_df.loc[key_idx, "p_value"].to_numpy())
        out_df.loc[key_idx, "q_value"] = qvals

    out_df["is_significant"] = (out_df["q_value"] <= args.q_thresh).astype(int)
    out_df.sort_values(["key", "sid", "electrode", "lag_index"], inplace=True)

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.outfile, index=False)

    print(
        "Saved stats to",
        args.outfile,
        "records=",
        len(out_df),
        "significant=",
        int(out_df["is_significant"].sum()),
        "missing_null_files=",
        missing_null,
        "missing_observed_patterns=",
        missing_obs,
    )


if __name__ == "__main__":
    main()
