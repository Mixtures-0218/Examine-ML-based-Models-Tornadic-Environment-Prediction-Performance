"""
2025 Case Study: 168h FB Bar Chart — ML Models + GFS
=====================================================
Similar to fig1_168h_fb_bars but for the single 2025-05-19 case,
including GFS (compared against GFS fct0, not ERA5).
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = '/Volumes/T7/Research/spatial_evaluation'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

TIME = '2025-05-19-23UTC'

models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi', 'gfs']
model_labels = {
    'pangu': 'Pangu',
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'fuxi': 'FuXi',
    'gfs': 'GFS'
}
model_colors = {
    'pangu': '#1f77b4',
    'fengwu_v1': '#8BCF8B',
    'fengwu_v2': '#42949E',
    'fuxi': '#d62728',
    'gfs': '#9B59B6'
}

params = ['cape', 'cin', 'k_idx', 'positive_helicity', 'negative_helicity', 'lcl', 'lfc', 'stp']
param_labels = {
    'cape': 'CAPE',
    'lcl': 'LCL',
    'lfc': 'LFC',
    'k_idx': 'K Index',
    'positive_helicity': 'SRH+',
    'negative_helicity': 'SRH-',
    'cin': 'CIN',
    'stp': 'STP'
}


def compute_fb(fcst, ref, param):
    """FB: area ratio of fcst vs ref above/below percentile threshold."""
    if param in ['cin', 'negative_helicity']:
        p_thresh = np.nanpercentile(ref, 25)
        n_ref = np.nansum((ref <= p_thresh).astype(float))
        n_fcst = np.nansum((fcst <= p_thresh).astype(float))
    else:
        p_thresh = np.nanpercentile(ref, 75)
        n_ref = np.nansum((ref >= p_thresh).astype(float))
        n_fcst = np.nansum((fcst >= p_thresh).astype(float))
    if n_ref == 0:
        return np.nan
    return n_fcst / n_ref


def compute_168h_fb():
    """Compute FB at 168h for all models (including GFS vs GFS fct0)."""
    fb_result = {m: {} for m in models}

    for param in params:
        # ERA5 reference (used by ML models)
        real_path = os.path.join(BASE_DIR, TIME, f'{param}_real.npy')
        if not os.path.exists(real_path):
            for m in models:
                fb_result[m][param] = np.nan
            continue
        era5 = np.load(real_path)

        # GFS reference (GFS fct0 = index 0)
        gfs_path = os.path.join(BASE_DIR, TIME, 'gfs', f'{param}_gfs.npy')
        gfs_data = None
        if os.path.exists(gfs_path):
            gfs_data = np.load(gfs_path)
            gfs_fct0 = gfs_data[0]  # analysis / f005 valid at event time

        for model in models:
            if model == 'gfs':
                if gfs_data is None:
                    fb_result[model][param] = np.nan
                    continue
                fcst_168 = gfs_data[5]  # f168
                fb = compute_fb(fcst_168, gfs_fct0, param)
            else:
                fcst_path = os.path.join(BASE_DIR, TIME, model, f'{param}_{model}.npy')
                if not os.path.exists(fcst_path):
                    fb_result[model][param] = np.nan
                    continue
                fcst_all = np.load(fcst_path)
                fcst_168 = fcst_all[4]  # 168h = index 4
                fb = compute_fb(fcst_168, era5, param)
            fb_result[model][param] = fb

    return fb_result


def plot_2025_fb_bars(fb_result):
    """Plot grouped bar chart: 2025 case 168h FB, 5 models × 8 params."""
    fig, ax = plt.subplots(figsize=(20, 9))

    x_pos = np.arange(len(params))
    n_models = len(models)
    width = 0.15
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

    for mi, model in enumerate(models):
        values = []
        for param in params:
            fb = fb_result[model].get(param, np.nan)
            values.append(fb)

        bars = ax.bar(
            x_pos + offsets[mi], values, width * 0.85,
            color=model_colors[model], label=model_labels[model],
            edgecolor='white', linewidth=0.5, alpha=0.88
        )

        # Annotate values on bars
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            color = 'red' if val > 1.2 else ('blue' if val < 0.5 else 'black')
            offset = 0.04 if val >= 0 else 0.02
            label = '0' if abs(val) < 0.005 else f'{val:.2f}'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                label,
                ha='center', fontsize=11, fontweight='bold', color=color
            )

    # Reference line at FB=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.8, alpha=0.6)

    # Shade zones
    ax.axhspan(0, 0.5, alpha=0.07, color='blue')
    ax.axhspan(1.5, 4.0, alpha=0.07, color='red')
    ax.text(len(params) - 0.5, 0.25, 'ERASED', fontsize=14, color='blue', alpha=0.5, fontweight='bold')
    ax.text(len(params) - 0.5, 3.0, 'OVER-PREDICT', fontsize=14, color='red', alpha=0.5, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([param_labels[p] for p in params], rotation=45, ha='right', fontsize=15)
    ax.set_ylabel('Frequency Bias (FB)', fontsize=17, fontweight='bold')
    ax.set_title(
        '2025-05-19 EF3+ Tornado Case — 168h Frequency Bias\n'
        'ML models: FB vs ERA5  |  GFS: FB vs GFS fct0 (analysis)',
        fontsize=20, fontweight='bold'
    )
    ax.set_ylim(-0.08, 4.2)
    ax.grid(True, alpha=0.25, axis='y')
    ax.legend(fontsize=14, loc='upper left', ncol=5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_2025_case_168h_fb.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    return fig


def print_summary(fb_result):
    """Print a markdown summary table."""
    print('\n## 2025 Case 168h FB Summary\n')
    header = '| Parameter | ' + ' | '.join(model_labels[m] for m in models) + ' |'
    sep = '|:----------|' + ':-----:|' * len(models)
    print(header)
    print(sep)

    for param in params:
        vals = []
        for m in models:
            v = fb_result[m].get(param, np.nan)
            if np.isnan(v):
                vals.append('N/A')
            else:
                vals.append(f'{v:.3f}')
        print(f'| {param_labels[param]} | ' + ' | '.join(vals) + ' |')

    print('\nML models: FB vs ERA5  |  GFS: FB vs GFS fct0')
    print('FB > 1.2 = over-prediction  |  FB < 0.5 = near-complete erasure')


if __name__ == '__main__':
    print('=' * 70)
    print('  2025 Case Study — 168h FB: ML Models + GFS')
    print('=' * 70)

    fb_result = compute_168h_fb()
    plot_2025_fb_bars(fb_result)
    print_summary(fb_result)

    print(f'\nDone. Output in {OUT_DIR}')

