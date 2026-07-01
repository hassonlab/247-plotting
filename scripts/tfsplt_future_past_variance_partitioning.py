import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/scripts')
import tfsplt_future_past_utils as pu
import warnings
warnings.filterwarnings('ignore')

# Configuration
res_d = "/scratch/gpfs/HASSON/ij9216/projects/code/247/247-encoding-dev/results/tfs"
thresh_joint = 0.01
output_dir = f"/scratch/gpfs/HASSON/ij9216/projects/code/247/247-plotting/results/roi_figures/future_past/variance_partitioning_{str(thresh_joint).split('.')[-1]}_thresh/"
# lags in data
lag_range_ms = 30000  # 30k range for these models

# Define lags (plot)
lag_start = -5000
lag_end = 5000
lag_step = 50
lags_to_plot = np.arange(lag_start, lag_end + lag_step, lag_step).tolist()

lags_to_plot_brain = [-2000, -1000, -500, 0, 500, 1000, 2000]


# Define colormaps for variance partition components
colormaps = {
    'unique_future': '#0D8D00',    # Future - green
    'unique_past': '#A90008',      # Past - red
    'unique_word': '#AC7000',      # Word - orange
    'shared_fut_pas': '#660099',   # Future/Past shared - purple
    'shared_fut_word': '#008080',  # Future/Word shared - teal
    'shared_pas_word': '#FF6B35',  # Past/Word shared - burnt orange
    'shared_three_way': '#FFD700'  # Three-way shared - gold
}

# Centralized order/mapping so all plots include every component
COMPONENT_KEYS = [
    'u_fut', 'u_pas', 'u_word',
    'c_fp', 'c_fw', 'c_pw', 'c_fpw'
]

COMPONENT_LABELS = {
    'u_fut': 'Unique Future',
    'u_pas': 'Unique Past',
    'u_word': 'Unique Word',
    'c_fp': 'Shared Future/Past',
    'c_fw': 'Shared Future/Word',
    'c_pw': 'Shared Past/Word',
    'c_fpw': 'Three-way Shared'
}

COMPONENT_COLORS = {
    'u_fut': 'unique_future',
    'u_pas': 'unique_past',
    'u_word': 'unique_word',
    'c_fp': 'shared_fut_pas',
    'c_fw': 'shared_fut_word',
    'c_pw': 'shared_pas_word',
    'c_fpw': 'shared_three_way'
}

# Distinct linestyles to make overlapping components visible even if numerically similar
# COMPONENT_STYLES = {
#     'u_fut': '-',
#     'u_pas': '-.',
#     'u_word': ':',
#     'c_fp': '--',
#     'c_fw': (0, (3, 1, 1, 1)),  # dash-dot-dot
#     'c_pw': (0, (5, 2)),
#     'c_fpw': (0, (1, 1))
# }
COMPONENT_STYLES = {
    'u_fut': '-',
    'u_pas': '-',
    'u_word': '-',
    'c_fp': '-',
    'c_fw': '-',
    'c_pw': '-',
    'c_fpw': '-'
}


# Subjects
subjects = ["625", "676", "717", "798"]

print(f"Results directory: {res_d}")
print(f"Output directory: {output_dir}")
print(f"Joint threshold: {thresh_joint}")
print(f"Lag range: {lag_start} to {lag_end} ms")
print(f"Number of lags: {len(lags_to_plot)}")


def load_partition_models(res_dir, mode='comp'):
    """
    Load the 7 variance partitioning models for a given mode ('comp' or 'prod').
    Returns a dictionary of dataframes.
    """
    print(f"Loading variance partitioning models for {mode}...")
    
    suffix_map = {
        'comp': 'comp',
        'prod': 'prod'
    }
    s = suffix_map[mode]
    
    models = {}
    
    # Helper to load
    def load(name):
        # path = f"{res_dir}/ij-tfs-%s-gpt2-xl-bandedRidge-lag30k-50-var-partition_pca300_{name}_{s}.csv"
        path = f"{res_dir}/ij-tfs-%s-mistral-mistral_bandedRidge-lag30k-50-var-partition_pca300_{name}_load-splits_no-shift_cnxt8_deltasv2_{s}.csv"
        df = pd.read_csv(path)
        return pu.add_roi_label_to_results(df)

    models['full'] = load("all")
    models['f']    = load("fut-only")
    models['p']    = load("pas-only")
    models['w']    = load("word-only")
    models['fp']   = load("fut-pas-no-word")
    # models['fw']   = load("fut-word-no-pas")
    # models['pw']   = load("pas-word-no-fut")
    models['fw']   = load("fut-word")
    models['pw']   = load("pas-word")
    
    print(f"Loaded {mode} models.")
    return models


import numpy as np
import pandas as pd

def compute_variance_partitions(full_df, fut_only_df, pas_only_df, word_only_df,
                                fut_pas_df, fut_word_df, pas_word_df):
    """
    Compute unique and shared variance fractions using R^2 (Coefficient of Determination).
    
    Partitions the total variance (R^2_Full) into 7 mutually exclusive components:
    - 3 Unique: U(F), U(P), U(W)
    - 3 Two-way Shared: C(FP), C(FW), C(PW)
    - 1 Three-way Shared: C(FPW)
    """

    def filter_label3(df):
        if 'label3' in df.columns:
            joint_rows = df[df['label3'] == 'joint']
            return joint_rows.copy() if len(joint_rows) > 0 else df[df['label3'].isna()].copy()
        return df.copy()
    
    # 1. Align and Filter
    dfs = [full_df, fut_only_df, pas_only_df, word_only_df, fut_pas_df, fut_word_df, pas_word_df]
    dfs = [filter_label3(df) for df in dfs]

    def lag_cols(df):
        return {c for c in df.columns if c.isdigit()}

    common_lags = set.intersection(*(lag_cols(df) for df in dfs))
    numeric_cols = sorted(common_lags, key=lambda x: int(x))
    
    key_cols = ['subject', 'electrode', 'roi']

    def renamed_block(df, suffix):
        renames = {c: f"{c}{suffix}" for c in numeric_cols}
        return df[key_cols + numeric_cols].rename(columns=renames)

    # 2. Merge all models into one wide dataframe
    suffixes = ["_full", "_f", "_p", "_w", "_fp", "_fw", "_pw"]
    merged = renamed_block(dfs[0], suffixes[0])
    for df, suff in zip(dfs[1:], suffixes[1:]):
        merged = merged.merge(renamed_block(df, suff), on=key_cols, how='inner')

    # 3. Extract R^2 values (SQUARING the Pearson r)
    def get_r2(suffix):
        # return merged[[f"{c}{suffix}" for c in numeric_cols]].values**2
        return np.maximum(merged[[f"{c}{suffix}" for c in numeric_cols]].values**2, 0)

    r2_full = get_r2("_full")
    r2_f    = get_r2("_f")
    r2_p    = get_r2("_p")
    r2_w    = get_r2("_w")
    r2_fp   = get_r2("_fp")
    r2_fw   = get_r2("_fw")
    r2_pw   = get_r2("_pw")

    # 4. Compute the 7 mutually exclusive commonality components via Möbius inversion.
    # Each component sums to the full-model R^2 when combined.
    # Reference: commonality analysis for three predictors.

    # Unique components
    u_f = r2_full - r2_pw
    u_p = r2_full - r2_fw
    u_w = r2_full - r2_fp

    # Two-way shared (Exclusive of the third variable)
    # Derived from: C_xy = R2_xz + R2_yz - R2_z - R2_xyz (where z is the 3rd var)
    # Alternatively: C_xy = R2_pairwise_with_z - R2_z - U_x (or U_y)
    
    # Shared Future/Past (z=Word)
    c_fp = r2_fw + r2_pw - r2_w - r2_full
    
    # Shared Future/Word (z=Past)
    c_fw = r2_fp + r2_pw - r2_p - r2_full
    
    # Shared Past/Word (z=Future)
    c_pw = r2_fp + r2_fw - r2_f - r2_full

    # Three-way shared component
    c_fpw = r2_full - r2_fp - r2_fw - r2_pw + r2_f + r2_p + r2_w

    # Optional: Noise floor handling (Common in ECoG/fMRI)
    # We allow negative values to detect suppression, but if you only care 
    # about magnitude, you can apply np.maximum(x, 0) here.

    
    # 5. Build Result Dataframe
    result_df = merged[key_cols].copy()
    for i, col in enumerate(numeric_cols):
        current_lag = -30000 + (int(col) * 50)
        prefix = f"{current_lag}"
        
        result_df[f'{prefix}_total_r2'] = r2_full[:, i]
        result_df[f'{prefix}_u_fut']    = u_f[:, i]
        result_df[f'{prefix}_u_pas']    = u_p[:, i]
        result_df[f'{prefix}_u_word']   = u_w[:, i]
        result_df[f'{prefix}_c_fp']     = c_fp[:, i]
        result_df[f'{prefix}_c_fw']     = c_fw[:, i]
        result_df[f'{prefix}_c_pw']     = c_pw[:, i]
        result_df[f'{prefix}_c_fpw']    = c_fpw[:, i]

    print(f"Aligned electrodes: {len(result_df)}")
    return result_df, numeric_cols




# Diagnostic helper: summarize component magnitudes over the lags we plot
def log_component_stats(df, label, lags):
    if df.empty:
        print(f"[DEBUG] {label}: no rows after filtering")
        return
    print(f"[DEBUG] {label}: component summary over {len(lags)} lags and {len(df)} electrodes")
    for comp_key in COMPONENT_KEYS:
        cols = [f"{lag}_{comp_key}" for lag in lags if f"{lag}_{comp_key}" in df.columns]
        if not cols:
            print(f"  {comp_key}: missing all plotted lag columns")
            continue
        vals = np.nan_to_num(df[cols].values.astype(float), nan=0.0)
        max_abs = np.max(np.abs(vals)) if vals.size else 0.0
        nonzero = np.sum(np.abs(vals) > 1e-6)
        print(f"  {comp_key}: max|val|={max_abs:.4g}, nonzero(>1e-6)={nonzero}/{vals.size}")




# for filtered electrodes, in lags_to_plot, compute and plot relative variance per electrode (as % of total variance), print all figures to a pdf
import os
from matplotlib.backends.backend_pdf import PdfPages

def plot_variance_partitions_lines_to_pdf(df, filename_suffix, lags, colormaps, output_dir):
    if df.empty:
        print(f"No electrodes passed filtering for {filename_suffix}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, f"variance_partitioning_lines_{filename_suffix}.pdf")

    print(f"Plotting {len(df)} electrodes for {filename_suffix} to {pdf_path}...")
    
    with PdfPages(pdf_path) as pdf:
        for idx, row in df.iterrows():
            subj, elec, roi = str(row.get('subject', '')), str(row.get('electrode', '')), str(row.get('roi', ''))
            electrode_id = f"{subj} {elec} {roi}"
            
            comp_data = {c: [] for c in COMPONENT_KEYS}
            total_r2_vals = []
            plot_lags = []

            for lag in lags:
                lag_str = str(lag)
                full_col = f"{lag_str}_total_r2"
                
                if full_col not in df.columns:
                    continue
                
                total_r2_vals.append(row[full_col])
                plot_lags.append(lag)

                for comp_key in COMPONENT_KEYS:
                    col_name = f"{lag_str}_{comp_key}"
                    comp_data[comp_key].append(row[col_name] if col_name in df.columns else 0.0)
            
            if not plot_lags:
                continue

            # Convert to arrays and replace NaNs with 0 so lines render
            comp_data = {k: np.nan_to_num(np.array(v, dtype=float), nan=0.0) for k, v in comp_data.items()}
            total_r2_vals = np.nan_to_num(np.array(total_r2_vals, dtype=float), nan=0.0)

            # Debug: report if specific components are entirely zero
            for dbg_key in ['c_fp', 'c_fw']:
                if dbg_key in comp_data:
                    max_abs_dbg = np.max(np.abs(comp_data[dbg_key])) if len(comp_data[dbg_key]) else 0
                    if np.allclose(comp_data[dbg_key], 0):
                        print(f"[DEBUG] {electrode_id}: component {dbg_key} all zeros across plotted lags ({len(plot_lags)} lags)")
                        # Check if columns existed for first plotted lag
                        example_lag = plot_lags[0]
                        col_name = f"{example_lag}_{dbg_key}"
                        if col_name not in df.columns:
                            print(f"  [DEBUG] Missing column {col_name} in dataframe")
                        else:
                            print(f"  [DEBUG] Example value at {example_lag} ms: {row[col_name]}")
                    elif max_abs_dbg < 1e-4:
                        print(f"[DEBUG] {electrode_id}: component {dbg_key} very small (max|val|={max_abs_dbg:.2e})")
                    elif np.allclose(comp_data[dbg_key], comp_data.get('c_pw', comp_data[dbg_key]), atol=1e-6):
                        print(f"[DEBUG] {electrode_id}: component {dbg_key} overlaps c_pw (allclose within 1e-6)")

            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the 7 components
            for comp_key in COMPONENT_KEYS:
                label = COMPONENT_LABELS[comp_key]
                color = colormaps.get(COMPONENT_COLORS[comp_key], '#333333')
                style = COMPONENT_STYLES.get(comp_key, '-')
                ax.plot(plot_lags, comp_data[comp_key], label=label, color=color, linewidth=1.5, linestyle=style)

            # Total R2
            ax.plot(plot_lags, total_r2_vals, label='Total Variance Explained ($R^2$)', color='black', linewidth=2, linestyle='-')
            
            ax.set_title(f"Variance Partitioning ($R^2$): {electrode_id}")
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("Variance Explained ($R^2$)")
            # Dynamic y-limits so negative shared terms are visible
            min_val = min([vals.min() if len(vals) else 0 for vals in comp_data.values()] + [total_r2_vals.min()])
            max_val = max([vals.max() if len(vals) else 0 for vals in comp_data.values()] + [total_r2_vals.max()])
            lower = min(min_val * 1.1, -0.02)
            upper = max(max_val * 1.1, 0.05)
            ax.set_ylim(lower, upper)
            
            ax.axvline(0, color='black', linestyle=':', alpha=0.5)
            ax.grid(True, linestyle=':', alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

from scipy.stats import sem

def plot_roi_variance_partitions_to_pdf(df, filename_suffix, lags, colormaps, output_dir):
    if df.empty:
        print(f"No data for ROI plots: {filename_suffix}")
        return

    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Handle None/NaN ROIs so sorted() doesn't crash
    df['roi'] = df['roi'].fillna('Unknown').astype(str)
    rois = sorted(df['roi'].unique())

    pdf_path = os.path.join(output_dir, f"variance_partitioning_ROI_SUMMARY_{filename_suffix}.pdf")
    
    print(f"Plotting {len(rois)} ROIs to {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        for roi in rois:
            roi_df = df[df['roi'] == roi]
            n_elec = len(roi_df)
            
            # Skip if no electrodes (shouldn't happen with unique(), but good safety)
            if n_elec == 0: continue

            fig, ax = plt.subplots(figsize=(12, 7))
            
            valid_lags = []
            # Plot individual components
            for comp_key in COMPONENT_KEYS:
                label = COMPONENT_LABELS[comp_key]
                means, sems = [], []
                current_valid_lags = []

                for lag in lags:
                    col = f"{lag}_{comp_key}"
                    if col in roi_df.columns:
                        data = np.nan_to_num(roi_df[col].values.astype(float), nan=0.0)
                        means.append(np.mean(data))
                        sems.append(sem(data) if len(data) > 1 else 0)
                        current_valid_lags.append(lag)
                
                if not current_valid_lags: continue
                valid_lags = current_valid_lags # synchronize x-axis

                color = colormaps.get(COMPONENT_COLORS[comp_key], '#333333')
                style = COMPONENT_STYLES.get(comp_key, '-')
                ax.plot(valid_lags, means, label=label, color=color, linewidth=2, linestyle=style)
                if n_elec > 1:
                    ax.fill_between(valid_lags, np.array(means) - np.array(sems), 
                                    np.array(means) + np.array(sems), color=color, alpha=0.15)

            # Plot Total R2 Mean
            total_means = [np.mean(np.nan_to_num(roi_df[f"{lag}_total_r2"].values.astype(float), nan=0.0)) for lag in valid_lags]
            ax.plot(valid_lags, total_means, label='Total $R^2$', color='black', linestyle='-', linewidth=1.5, alpha=0.6)

            ax.set_title(f"ROI: {roi} (N = {n_elec} electrodes)", fontsize=14)
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("Mean Variance Explained ($R^2$)")
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"ROI summary for {filename_suffix} finished.")

def run_variance_partitioning_pipeline():
    """
    To be called from other scripts.
    Loads models, computes variance partitions, filters, and returns the filtered dataframes.
    """
    comp_models = load_partition_models(res_d, "comp")
    prod_models = load_partition_models(res_d, "prod")

    comp_partitions, numeric_cols = compute_variance_partitions(
        comp_models['full'], comp_models['f'], comp_models['p'], comp_models['w'],
        comp_models['fp'], comp_models['fw'], comp_models['pw']
    )
    prod_partitions, _ = compute_variance_partitions(
        prod_models['full'], prod_models['f'], prod_models['p'], prod_models['w'],
        prod_models['fp'], prod_models['fw'], prod_models['pw']
    )

    full_r_cols = [col for col in comp_partitions.columns if col.endswith('_total_r2')]
    mask_comp = comp_partitions[full_r_cols].max(axis=1) >= thresh_joint
    comp_partitions_filtered = comp_partitions[mask_comp].copy()
    
    mask_prod = prod_partitions[full_r_cols].max(axis=1) >= thresh_joint
    prod_partitions_filtered = prod_partitions[mask_prod].copy()

    return comp_partitions_filtered, prod_partitions_filtered


def main():
    # Load Models
    comp_models = load_partition_models(res_d, "comp")
    prod_models = load_partition_models(res_d, "prod")

    # Compute Partitions
    print("Computing variance partitions for comprehension...")
    comp_partitions, numeric_cols = compute_variance_partitions(
        comp_models['full'], comp_models['f'], comp_models['p'], comp_models['w'],
        comp_models['fp'], comp_models['fw'], comp_models['pw']
    )
    
    print("Computing variance partitions for production...")
    prod_partitions, _ = compute_variance_partitions(
        prod_models['full'], prod_models['f'], prod_models['p'], prod_models['w'],
        prod_models['fp'], prod_models['fw'], prod_models['pw']
    )
    
    print(f"\nComprehension partitions shape: {comp_partitions.shape}")
    print(f"Production partitions shape: {prod_partitions.shape}")
    print(f"Number of lag columns: {len(numeric_cols)}")

    # Filter
    full_r_cols = [col for col in comp_partitions.columns if col.endswith('_total_r2')]
    mask_comp = comp_partitions[full_r_cols].max(axis=1) >= thresh_joint
    comp_partitions_filtered = comp_partitions[mask_comp].copy()
    
    mask_prod = prod_partitions[full_r_cols].max(axis=1) >= thresh_joint
    prod_partitions_filtered = prod_partitions[mask_prod].copy()

    # Log Stats
    log_component_stats(comp_partitions_filtered, "comp", lags_to_plot)
    log_component_stats(prod_partitions_filtered, "prod", lags_to_plot)

    # Plot
    print('Generating plots...')
    plot_roi_variance_partitions_to_pdf(comp_partitions_filtered, "comp", lags_to_plot, colormaps, output_dir)
    plot_roi_variance_partitions_to_pdf(prod_partitions_filtered, "prod", lags_to_plot, colormaps, output_dir)
    plot_variance_partitions_lines_to_pdf(comp_partitions_filtered, "comp", lags_to_plot, colormaps, output_dir)
    plot_variance_partitions_lines_to_pdf(prod_partitions_filtered, "prod", lags_to_plot, colormaps, output_dir)

if __name__ == "__main__":
    main()
