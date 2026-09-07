"""
Fengwu v1 vs v2 analysis: Generate all figures (English labels only)

Figure list:
  fig1_crossvar_delta_csi.png    - Cross-variable CSI change (v2 - v1)
  fig2_cape_csi_all_models.png   - CAPE CSI across lead times, all models
  fig3_cape_std_ratio.png        - CAPE std ratio v1 vs v2
  fig4_delta_std_vs_delta_csi.png- Scatter: Dstd_r vs DCSI (all variables)
  fig5_cape_per_case_csi.png     - Per-case CAPE CSI heatmap
  fig6_cape_fb_evolution.png     - CAPE FB across lead times
  fig7_cape_mode_interest.png    - CAPE MODE interest score
  fig8_vertical_levels_concept.png - Vertical levels involved per variable
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd

# ============================================================
# Configuration
# ============================================================
BASE_DIR = '/Volumes/T7/Research/spatial_evaluation'
OUTPUT_DIR = os.path.join(BASE_DIR, 'research/Fengwu_analysis')

models = ['fengwu_v1', 'fengwu_v2', 'pangu', 'fuxi']
model_labels = {
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'pangu': 'Pangu',
    'fuxi': 'FuXi'
}
model_colors = {
    'fengwu_v1': '#8BCF8B',
    'fengwu_v2': '#42949E',
    'pangu': '#1f77b4',
    'fuxi': '#d62728'
}
model_markers = {
    'fengwu_v1': 's',
    'fengwu_v2': '^',
    'pangu': 'o',
    'fuxi': 'D'
}

lead_names = ['24h', '48h', '72h', '120h', '168h']
lead_hours = [24, 48, 72, 120, 168]
lead_indices = [0, 1, 2, 3, 4]

times = [
    '2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
    '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
    '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
    '2025-05-19-23UTC'
]

parameters = {
    'CAPE': 'cape',
    'CIN': 'cin',
    'STP': 'stp',
    'K Index': 'k_idx',
    'LCL': 'lcl',
    'LFC': 'lfc',
    'SRH+': 'positive_helicity',
    'SRH-': 'negative_helicity'
}

params_for_scatter = [
    ('CAPE', 'cape'),
    ('CIN', 'cin'),
    ('STP', 'stp'),
    ('K Index', 'k_idx'),
    ('LCL', 'lcl'),
    ('LFC', 'lfc'),
    ('SRH+', 'positive_helicity'),
    ('SRH-', 'negative_helicity')
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 15,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.weight': 'normal'
})


# ============================================================
# Helper: load CSI data
# ============================================================
def load_csi(model, param_key, lead_idx):
    path = os.path.join(BASE_DIR, f'avg_scores/{model}/scores_meanvar_{param_key}_{model}.npz')
    if os.path.exists(path):
        data = np.load(path)
        return data['mean'][lead_idx, 1]
    return np.nan


def load_pod(model, param_key, lead_idx):
    path = os.path.join(BASE_DIR, f'avg_scores/{model}/scores_meanvar_{param_key}_{model}.npz')
    if os.path.exists(path):
        data = np.load(path)
        return data['mean'][lead_idx, 0]
    return np.nan


def load_fss(model, param_key, lead_idx):
    path = os.path.join(BASE_DIR, f'avg_scores/{model}/scores_meanvar_{param_key}_{model}.npz')
    if os.path.exists(path):
        data = np.load(path)
        return data['mean'][lead_idx, 2]
    return np.nan


# ============================================================
# Fig 1: Cross-variable CSI change (v2 - v1) at 24h and 72h
# ============================================================
def fig1_crossvar_delta_csi():
    print('Plotting Fig 1: cross-variable CSI change...')

    param_names = []
    delta_24h = []
    delta_72h = []
    delta_120h = []

    for pname, pkey in parameters.items():
        v1_24 = load_csi('fengwu_v1', pkey, 0)
        v2_24 = load_csi('fengwu_v2', pkey, 0)
        v1_72 = load_csi('fengwu_v1', pkey, 2)
        v2_72 = load_csi('fengwu_v2', pkey, 2)
        v1_120 = load_csi('fengwu_v1', pkey, 3)
        v2_120 = load_csi('fengwu_v2', pkey, 3)

        param_names.append(pname)
        delta_24h.append(v2_24 - v1_24)
        delta_72h.append(v2_72 - v1_72)
        delta_120h.append(v2_120 - v1_120)

    x = np.arange(len(param_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width, delta_24h, width, label='24h',
                   color='#5dade2', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, delta_72h, width, label='72h',
                   color='#f5b041', edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, delta_120h, width, label='120h',
                   color='#e74c3c', edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Variable', fontweight='bold')
    ax.set_ylabel('CSI Change (v2 - v1)', fontweight='bold')
    ax.set_title('FengWu v2 vs v1: Cross-Variable CSI Change', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=30, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                offset = 0.002
            else:
                va = 'top'
                offset = -0.002
            ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                    f'{height:+.3f}', ha='center', va=va, fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_crossvar_delta_csi.png'))
    plt.close()
    print('  -> fig1_crossvar_delta_csi.png')


# ============================================================
# Fig 2: CAPE CSI across lead times — all 4 models
# ============================================================
def fig2_cape_csi_all_models():
    print('Plotting Fig 2: CAPE CSI all models...')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: CSI vs lead time
    for model in models:
        csi_vals = []
        for li in lead_indices:
            csi = load_csi(model, 'cape', li)
            csi_vals.append(csi)
        ax1.plot(lead_hours, csi_vals, marker=model_markers[model],
                 color=model_colors[model], label=model_labels[model],
                 linewidth=2, markersize=8)

    ax1.set_xlabel('Lead Time (hours)', fontweight='bold')
    ax1.set_ylabel('CSI (Critical Success Index)', fontweight='bold')
    ax1.set_title('CAPE CSI Across Lead Times', fontweight='bold')
    ax1.set_xticks(lead_hours)
    ax1.set_xticklabels(lead_names)
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.55)

    # Right: FSS vs lead time
    for model in models:
        fss_vals = []
        for li in lead_indices:
            fss = load_fss(model, 'cape', li)
            fss_vals.append(fss)
        ax2.plot(lead_hours, fss_vals, marker=model_markers[model],
                 color=model_colors[model], label=model_labels[model],
                 linewidth=2, markersize=8)

    ax2.set_xlabel('Lead Time (hours)', fontweight='bold')
    ax2.set_ylabel('FSS (Fractions Skill Score)', fontweight='bold')
    ax2.set_title('CAPE FSS Across Lead Times', fontweight='bold')
    ax2.set_xticks(lead_hours)
    ax2.set_xticklabels(lead_names)
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_cape_csi_all_models.png'))
    plt.close()
    print('  -> fig2_cape_csi_all_models.png')


# ============================================================
# Fig 3: CAPE std ratio: v1 vs v2 across lead times
# ============================================================
def fig3_cape_std_ratio():
    print('Plotting Fig 3: CAPE std ratio...')

    v1_means = []
    v2_means = []
    v1_stds = []
    v2_stds = []

    for li in lead_indices:
        v1_ratios = []
        v2_ratios = []
        for time in times:
            real_path = os.path.join(BASE_DIR, time, 'cape_real.npy')
            if not os.path.exists(real_path):
                continue
            real = np.load(real_path)
            real_std = np.nanstd(real)
            if real_std == 0:
                continue
            for model, ratios in zip(['fengwu_v1', 'fengwu_v2'], [v1_ratios, v2_ratios]):
                path = os.path.join(BASE_DIR, time, model, f'cape_{model}.npy')
                if not os.path.exists(path):
                    continue
                fcst = np.load(path)[li]
                fcst_std = np.nanstd(fcst)
                ratios.append(fcst_std / real_std)

        v1_means.append(np.mean(v1_ratios))
        v2_means.append(np.mean(v2_ratios))
        v1_stds.append(np.std(v1_ratios))
        v2_stds.append(np.std(v2_ratios))

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(lead_names))
    width = 0.3

    bars1 = ax.bar(x - width/2, v1_means, width, yerr=v1_stds,
                   label='FengWu v1', color='#8BCF8B', capsize=4,
                   edgecolor='white', linewidth=0.5, error_kw={'elinewidth': 1.5})
    bars2 = ax.bar(x + width/2, v2_means, width, yerr=v2_stds,
                   label='FengWu v2', color='#42949E', capsize=4,
                   edgecolor='white', linewidth=0.5, error_kw={'elinewidth': 1.5})

    ax.axhline(y=1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.7, label='Perfect (ratio=1)')
    ax.set_xlabel('Lead Time', fontweight='bold')
    ax.set_ylabel('Spatial Std Ratio (forecast / observation)', fontweight='bold')
    ax.set_title('CAPE Variance Preservation: FengWu v1 vs v2', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lead_names)
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (v1, v2) in enumerate(zip(v1_means, v2_means)):
        ax.text(i - width/2, v1 + v1_stds[i] + 0.02, f'{v1:.3f}',
                ha='center', fontsize=13)
        ax.text(i + width/2, v2 + v2_stds[i] + 0.02, f'{v2:.3f}',
                ha='center', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_cape_std_ratio.png'))
    plt.close()
    print('  -> fig3_cape_std_ratio.png')


# ============================================================
# Fig 4: Scatter: std_ratio change vs CSI change (all variables, 72h)
# ============================================================
def fig4_delta_std_vs_delta_csi():
    print('Plotting Fig 4: std ratio vs CSI scatter...')

    delta_std = []
    delta_csi = []
    labels = []

    for pname, pkey in params_for_scatter:
        v1_ratios, v2_ratios = [], []
        for time in times:
            real_path = os.path.join(BASE_DIR, time, f'{pkey}_real.npy')
            if not os.path.exists(real_path):
                continue
            real = np.load(real_path)
            real_std = np.nanstd(real)
            if real_std == 0:
                continue
            for model, ratios in zip(['fengwu_v1', 'fengwu_v2'], [v1_ratios, v2_ratios]):
                path = os.path.join(BASE_DIR, time, model, f'{pkey}_{model}.npy')
                if not os.path.exists(path):
                    continue
                fcst = np.load(path)[2]
                fcst_std = np.nanstd(fcst)
                ratios.append(fcst_std / real_std)

        if not v1_ratios or not v2_ratios:
            continue

        dsr = np.mean(v2_ratios) - np.mean(v1_ratios)
        d1 = np.load(os.path.join(BASE_DIR, f'avg_scores/fengwu_v1/scores_meanvar_{pkey}_fengwu_v1.npz'))
        d2 = np.load(os.path.join(BASE_DIR, f'avg_scores/fengwu_v2/scores_meanvar_{pkey}_fengwu_v2.npz'))
        dcsi = d2['mean'][2, 1] - d1['mean'][2, 1]

        delta_std.append(dsr)
        delta_csi.append(dcsi)
        labels.append(pname)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#e74c3c' if l == 'CAPE' else '#5dade2' for l in labels]
    sizes = [180 if l == 'CAPE' else 80 for l in labels]
    edgecolors = ['#c0392b' if l == 'CAPE' else '#2e86c1' for l in labels]

    for i in range(len(labels)):
        ax.scatter(delta_std[i], delta_csi[i], s=sizes[i], c=colors[i],
                   edgecolors=edgecolors[i], linewidth=2 if labels[i] == 'CAPE' else 1,
                   alpha=0.8, zorder=5)
        offset_x = 0.003
        offset_y = 0.005
        ax.annotate(labels[i], (delta_std[i] + offset_x, delta_csi[i] + offset_y),
                    fontsize=12 if labels[i] == 'CAPE' else 10,
                    fontweight='bold' if labels[i] == 'CAPE' else 'normal')

    # Add regression line
    coeffs = np.polyfit(delta_std, delta_csi, 1)
    x_fit = np.linspace(min(delta_std) - 0.02, max(delta_std) + 0.02, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.5, linewidth=1.5,
            label=f'Linear fit (slope={coeffs[0]:.2f})')

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax.set_xlabel('Std Ratio Change at 72h (v2 - v1)', fontweight='bold')
    ax.set_ylabel('CSI Change at 72h (v2 - v1)', fontweight='bold')
    ax.set_title('Variance Preservation Drives CSI Improvement\n(72h, All Variables)', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_delta_std_vs_delta_csi.png'))
    plt.close()
    print('  -> fig4_delta_std_vs_delta_csi.png')


# ============================================================
# Fig 5: Per-case CAPE CSI: v1 vs v2 heatmap
# ============================================================
def fig5_cape_per_case_csi():
    print('Plotting Fig 5: per-case CAPE CSI...')

    n_cases = len(times)
    n_leads = 3

    v1_matrix = np.full((n_cases, n_leads), np.nan)
    v2_matrix = np.full((n_cases, n_leads), np.nan)

    for li in range(n_leads):
        for ci, time in enumerate(times):
            for model, matrix in zip(['fengwu_v1', 'fengwu_v2'], [v1_matrix, v2_matrix]):
                real = np.load(os.path.join(BASE_DIR, time, 'cape_real.npy'))
                real_p75 = np.nanpercentile(real, 75)
                path = os.path.join(BASE_DIR, time, model, f'cape_{model}.npy')
                if not os.path.exists(path):
                    continue
                fcst = np.load(path)[li]
                hits = np.nansum((fcst >= real_p75) & (real >= real_p75))
                misses = np.nansum((fcst < real_p75) & (real >= real_p75))
                fas = np.nansum((fcst >= real_p75) & (real < real_p75))
                denom = hits + misses + fas
                if denom > 0:
                    matrix[ci, li] = hits / denom

    case_labels = [t.replace('-20UTC', '').replace('-23UTC', '').replace('-21UTC', '').replace('-22UTC', '')
                   for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    vmin, vmax = 0, 0.7

    im1 = ax1.imshow(v1_matrix, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(n_leads))
    ax1.set_xticklabels(lead_names[:n_leads])
    ax1.set_yticks(range(n_cases))
    ax1.set_yticklabels(case_labels)
    ax1.set_title('FengWu v1: CAPE CSI per Case', fontweight='bold')
    ax1.set_xlabel('Lead Time')
    ax1.set_ylabel('Case Date')

    for i in range(n_cases):
        for j in range(n_leads):
            val = v1_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.35 else 'black'
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

    im2 = ax2.imshow(v2_matrix, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(n_leads))
    ax2.set_xticklabels(lead_names[:n_leads])
    ax2.set_yticks(range(n_cases))
    ax2.set_yticklabels(case_labels)
    ax2.set_title('FengWu v2: CAPE CSI per Case', fontweight='bold')
    ax2.set_xlabel('Lead Time')
    ax2.set_ylabel('Case Date')

    for i in range(n_cases):
        for j in range(n_leads):
            val = v2_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.35 else 'black'
                ax2.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('CSI', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cape_per_case_csi.png'))
    plt.close()
    print('  -> fig5_cape_per_case_csi.png')


# ============================================================
# Fig 6: CAPE Frequency Bias (FB) evolution
# ============================================================
def fig6_cape_fb_evolution():
    print('Plotting Fig 6: CAPE FB evolution...')

    fig, ax = plt.subplots(figsize=(12, 8))

    for model in ['fengwu_v1', 'fengwu_v2', 'pangu', 'fuxi']:
        fb_means = []
        fb_stds = []
        for li in range(3):
            fb_vals = []
            for time in times:
                real = np.load(os.path.join(BASE_DIR, time, 'cape_real.npy'))
                real_p75 = np.nanpercentile(real, 75)
                n_real_high = np.nansum((real >= real_p75).astype(float))
                if n_real_high == 0:
                    n_real_high = 1

                path = os.path.join(BASE_DIR, time, model, f'cape_{model}.npy')
                if not os.path.exists(path):
                    continue
                fcst = np.load(path)[li]
                n_fcst_high = np.nansum((fcst >= real_p75).astype(float))
                fb_vals.append(n_fcst_high / n_real_high)

            fb_means.append(np.mean(fb_vals))
            fb_stds.append(np.std(fb_vals))

        ax.errorbar([24, 48, 72], fb_means, yerr=fb_stds,
                     marker=model_markers[model], color=model_colors[model],
                     label=model_labels[model], linewidth=2, markersize=8,
                     capsize=4, capthick=1.5)

    ax.axhline(y=1.0, color='gray', linewidth=1.5, linestyle='--', alpha=0.7,
               label='Perfect (FB=1)')
    ax.set_xlabel('Lead Time (hours)', fontweight='bold')
    ax.set_ylabel('Frequency Bias (FB)', fontweight='bold')
    ax.set_title('CAPE Frequency Bias Evolution', fontweight='bold')
    ax.set_xticks([24, 48, 72])
    ax.set_xticklabels(['24h', '48h', '72h'])
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_cape_fb_evolution.png'))
    plt.close()
    print('  -> fig6_cape_fb_evolution.png')


# ============================================================
# Fig 7: CAPE MODE interest scores — all models
# ============================================================
def fig7_cape_mode_interest():
    print('Plotting Fig 7: CAPE MODE interest...')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model in models:
        path = os.path.join(BASE_DIR, f'avg_mode/{model}/mode_summary_cape_{model}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        interests = df['interest_mean'].values
        matches = df['matches_mean'].values

        ax1.plot(lead_hours, interests, marker=model_markers[model],
                 color=model_colors[model], label=model_labels[model],
                 linewidth=2, markersize=8)

        ax2.plot(lead_hours, matches, marker=model_markers[model],
                 color=model_colors[model], label=model_labels[model],
                 linewidth=2, markersize=8)

    ax1.set_xlabel('Lead Time (hours)', fontweight='bold')
    ax1.set_ylabel('MODE Interest Score', fontweight='bold')
    ax1.set_title('CAPE MODE Interest Score', fontweight='bold')
    ax1.set_xticks(lead_hours)
    ax1.set_xticklabels(lead_names)
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)

    ax2.set_xlabel('Lead Time (hours)', fontweight='bold')
    ax2.set_ylabel('MODE Matched Objects', fontweight='bold')
    ax2.set_title('CAPE MODE Object Matches', fontweight='bold')
    ax2.set_xticks(lead_hours)
    ax2.set_xticklabels(lead_names)
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_cape_mode_interest.png'))
    plt.close()
    print('  -> fig7_cape_mode_interest.png')


# ============================================================
# Fig 8: Vertical levels involved per variable (conceptual diagram)
# ============================================================
def fig8_vertical_levels_concept():
    print('Plotting Fig 8: vertical levels concept...')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: variables involved vs delta CSI
    var_names = ['CAPE', 'LCL', 'LFC', 'K Index', 'CIN', 'STP', 'SRH+', 'SRH-']
    n_levels = [25, 1, 1, 3, 8, 25, 6, 6]
    deltas_72h = []
    for pname, pkey in params_for_scatter:
        v1 = load_csi('fengwu_v1', pkey, 2)
        v2 = load_csi('fengwu_v2', pkey, 2)
        deltas_72h.append(v2 - v1)

    color_map = ['#e74c3c' if d > 0.04 else '#f5b041' if d > 0 else
                 '#5dade2' if d > -0.04 else '#2980b9' for d in deltas_72h]

    bars = ax1.barh(var_names, n_levels, color=color_map, edgecolor='white', linewidth=0.5)

    for i, (bar, d) in enumerate(zip(bars, deltas_72h)):
        label = f'ΔCSI={d:+.3f}'
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 label, va='center', fontsize=11, fontweight='bold' if var_names[i] == 'CAPE' else 'normal')

    ax1.set_xlabel('Number of Vertical Levels Involved', fontweight='bold')
    ax1.set_title('Variables by Vertical Integration Depth\n(color = ΔCSI at 72h)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Right panel: schematic of the hypothesis
    ax2.axis('off')

    # Draw the schematic
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Title
    ax2.text(5, 9.5, 'Why CAPE Benefits Most from Vertical Improvements',
             ha='center', fontsize=13, fontweight='bold')

    # Draw three column labels
    ax2.text(1.5, 8.2, 'CAPE\n(20+ levels)', ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor='#1abc9c'))
    ax2.text(5, 8.2, 'LCL\n(surface only)', ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#fef9e7', edgecolor='#f1c40f'))
    ax2.text(8.5, 8.2, 'K Index\n(3 levels)', ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#fdedec', edgecolor='#e74c3c'))

    # Draw levels for CAPE
    y_start = 6.0
    for i in range(8):
        y = y_start - i * 0.6
        ax2.add_patch(plt.Rectangle((0.7, y), 1.6, 0.5, facecolor='#1abc9c', alpha=0.3 + i*0.08,
                                     edgecolor='#16a085', linewidth=0.5))
        if i == 0:
            ax2.text(1.5, y + 0.15, '200 hPa', ha='center', fontsize=7)
        elif i == 7:
            ax2.text(1.5, y + 0.15, 'Surface', ha='center', fontsize=7)

    # Levels for LCL
    ax2.add_patch(plt.Rectangle((4.0, 3.0), 2.0, 1.5, facecolor='#f1c40f', alpha=0.4,
                                 edgecolor='#f39c12', linewidth=1))
    ax2.text(5, 3.75, 'Surface\nT, Td', ha='center', fontsize=10, fontweight='bold')

    # Levels for K Index
    pressure_levels = ['850 hPa', '700 hPa', '500 hPa']
    for i, pl in enumerate(pressure_levels):
        y = 4.0 - i * 1.2
        ax2.add_patch(plt.Rectangle((7.5, y), 2.0, 0.8, facecolor='#e74c3c', alpha=0.3 + i*0.1,
                                     edgecolor='#c0392b', linewidth=0.5))
        ax2.text(8.5, y + 0.3, pl, ha='center', fontsize=13)

    # Delta markers
    ax2.text(1.5, 1.5, 'ΔCSI = +0.070\nΔStd_r = +0.119',
             ha='center', fontsize=11, fontweight='bold', color='#1abc9c',
             bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor='#1abc9c'))
    ax2.text(5, 1.5, 'ΔCSI = +0.050\nΔStd_r = +0.043',
             ha='center', fontsize=10, color='#f39c12',
             bbox=dict(boxstyle='round', facecolor='#fef9e7', edgecolor='#f1c40f'))
    ax2.text(8.5, 1.5, 'ΔCSI = -0.091\nΔStd_r = -0.031',
             ha='center', fontsize=10, fontweight='bold', color='#e74c3c',
             bbox=dict(boxstyle='round', facecolor='#fdedec', edgecolor='#e74c3c'))

    # Arrow explaining
    ax2.annotate('', xy=(3.5, 4.5), xytext=(1.5, 4.5),
                 arrowprops=dict(arrowstyle='->', color='#1abc9c', lw=2))
    ax2.text(2.5, 4.8, 'Vertical integration\namplifies improvement', ha='center', fontsize=9, color='#1abc9c')

    ax2.set_title('Conceptual: Vertical Integration Amplifier', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_vertical_levels_concept.png'))
    plt.close()
    print('  -> fig8_vertical_levels_concept.png')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('Generating Fengwu v1 vs v2 analysis figures...')
    print(f'Output: {OUTPUT_DIR}')
    print()

    fig1_crossvar_delta_csi()
    fig2_cape_csi_all_models()
    fig3_cape_std_ratio()
    fig4_delta_std_vs_delta_csi()
    fig5_cape_per_case_csi()
    fig6_cape_fb_evolution()
    fig7_cape_mode_interest()
    fig8_vertical_levels_concept()

    print('\nAll figures generated!')
