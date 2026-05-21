#!/usr/bin/env python3
"""
Multi-scale FSS Analysis Script
====================================
Evaluates Fractions Skill Score (FSS) at different neighborhood scales for
10 EF3 tornado events, and determines the optimal scale based on the
FSS_random(n) reference line from Antonio & Aitchison (2025).

Outputs:
  - results/fss_multiscale_mean.npz  — Cross-event mean FSS(n) and FSS_random(n)
  - results/summary.csv              — Optimal scale per (model, param, lead_time)
  - results/optimal_scale_counts.csv — Frequency statistics of each scale's "win"
  - plots/ scale curves per parameter/model
"""

import numpy as np
import os
import warnings
from scipy.ndimage import uniform_filter
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Path Configuration ---
DATA_ROOT = Path('/Volumes/T7/Research/spatial_evaluation')
OUTPUT_DIR = DATA_ROOT / 'fss_scale_analysis'
RESULTS_DIR = OUTPUT_DIR / 'results'
PLOTS_DIR = OUTPUT_DIR / 'plots'

for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Data Configuration ---
# All EF3 events
TIMES = [
    '2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
    '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
    '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
    '2025-05-19-23UTC'
]

MODELS = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']

PARAMETERS = [
    'cape', 'cin', 'positive_helicity', 'negative_helicity',
    'lcl', 'k_idx', 'lfc', 'ehi'
]

LEAD_TIMES = np.array([24, 48, 72, 120, 168])  # hours

# Neighborhood scales to evaluate (uniform_filter size = full window width)
# Note: grid 0.25 degrees, 21x25 points, each grid point ~25km
NEIGH_SIZES = np.array([3, 5, 7, 11])

# Corresponding physical scales (~km, approximate)
NEIGH_SIZE_KM = {
    1: 25,   # Grid scale (reference only)
    3: 75,   # n=1, window 3x3
    5: 125,  # n=2, window 5x5
    7: 175,  # n=3, window 7x7
    11: 275, # n=5, window 11x11
}

# Parameter display names
PARAM_LABELS = {
    'cape': 'CAPE',
    'cin': 'CIN',
    'positive_helicity': 'Pos SRH',
    'negative_helicity': 'Neg SRH',
    'lcl': 'LCL',
    'k_idx': 'K Index',
    'lfc': 'LFC',
    'ehi': 'EHI',
}

MODEL_LABELS = {
    'pangu': 'Pangu',
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'fuxi': 'FuXi',
}

MODEL_COLORS = {
    'pangu': '#1f77b4',
    'fengwu_v1': '#2e8b57',
    'fengwu_v2': '#3cb371',
    'fuxi': '#d62728',
}


def compute_fss(obs_evt, pred_evt, neigh_size):
    """Compute Fractions Skill Score at a single scale."""
    f_obs = uniform_filter(obs_evt.astype(float), size=neigh_size, mode='nearest')
    f_pred = uniform_filter(pred_evt.astype(float), size=neigh_size, mode='nearest')
    denom = np.sum(f_pred ** 2 + f_obs ** 2)
    if denom > 0:
        return 1 - np.sum((f_pred - f_obs) ** 2) / denom
    return np.nan


def compute_fss_random(obs_evt, neigh_size):
    """Compute FSS_random(n) reference line (Antonio & Aitchison 2025).

    FSS_random(n) = 2 mu_o,n^2 / (2 mu_o,n^2 + 1/(2n+1)^2 * mu_o,0(1-mu_o,0) + sigma_o,n^2)

    where n = (neigh_size - 1) / 2 is the half-width.
    """
    n_half = (neigh_size - 1) // 2  # Half-width (n in the paper)
    mu_o0 = np.mean(obs_evt)        # Base rate mu_o,0

    if mu_o0 == 0:
        return 0.0

    f_obs = uniform_filter(obs_evt.astype(float), size=neigh_size, mode='nearest')
    mu_on = np.mean(f_obs)          # mu_o,n (should equal mu_o,0)
    var_on = np.var(f_obs)          # sigma^2_o,n

    # Number of neighborhood cells
    n_cells = (2 * n_half + 1) ** 2

    numerator = 2 * mu_on ** 2
    denominator = numerator + (1 / n_cells) * mu_o0 * (1 - mu_o0) + var_on

    if denominator > 0:
        return numerator / denominator
    return np.nan


def compute_fss_random_mc(obs_evt, neigh_size, n_perm=1000, seed=42):
    """Monte Carlo validation of FSS_random(n) formula (diagnostic).

    Generates n_perm Bernoulli random fields (base rate = mu_o,0),
    smooths with uniform_filter, and computes mean FSS against the observation.
    """
    rng = np.random.default_rng(seed)
    mu_o0 = np.mean(obs_evt)
    shape = obs_evt.shape

    fss_vals = []
    for _ in range(n_perm):
        rand_field = rng.random(shape) < mu_o0
        f_rand = uniform_filter(rand_field.astype(float), size=neigh_size, mode='nearest')
        f_obs = uniform_filter(obs_evt.astype(float), size=neigh_size, mode='nearest')
        denom = np.sum(f_rand ** 2 + f_obs ** 2)
        if denom > 0:
            fss_vals.append(1 - np.sum((f_rand - f_obs) ** 2) / denom)

    return np.mean(fss_vals) if fss_vals else np.nan


def get_threshold(time, parameter, model):
    """Get parameter threshold (same as get_tornado_thresholds in Scores.py)."""
    root = DATA_ROOT / time
    if model != 'gfs':
        real_path = root / f'{parameter}_real.npy'
        real = np.load(str(real_path))
    else:
        real_path = root / 'gfs' / f'{parameter}_gfs.npy'
        real = np.load(str(real_path))[0]

    if parameter in ('negative_helicity', 'cin'):
        thresh = np.nanpercentile(real, 25)
    else:
        thresh = np.nanpercentile(real, 75)
    return thresh


def load_event_data(time, model, parameter):
    """Load observation and forecast data for a single event."""
    root = DATA_ROOT / time
    # Forecast
    pred_path = root / model / f'{parameter}_{model}.npy'
    preds = np.load(str(pred_path))  # shape (5, 21, 25)

    # Observation
    if model != 'gfs':
        truth_path = root / f'{parameter}_real.npy'
    else:
        truth_path = root / 'gfs' / f'{parameter}_gfs.npy'
    truth = np.load(str(truth_path))
    if model == 'gfs':
        truth = truth[0]  # GFS first dimension is members

    return truth, preds


def main():
    print('=' * 70)
    print('  Multi-scale FSS Analysis -- Based on FSS_random(n) reference line')
    print('  Antonio & Aitchison (2025)')
    print('=' * 70)

    n_sizes = len(NEIGH_SIZES)
    n_leads = len(LEAD_TIMES)
    n_events = len(TIMES)
    n_models = len(MODELS)
    n_params = len(PARAMETERS)

    # Store all results: { (model, param): { 'fss': (n_events, 5, n_sizes),
    #                                      'fss_random': (n_events, 5, n_sizes) } }
    all_results = {}

    for model in MODELS:
        for param in PARAMETERS:
            key = (model, param)
            event_fss = []
            event_fss_random = []
            n_loaded = 0

            for time in TIMES:
                try:
                    truth, preds = load_event_data(time, model, param)
                    thresh = get_threshold(time, param, model)
                except (FileNotFoundError, OSError) as e:
                    print(f'  [Skip] {time} {model} {param}: {e}')
                    continue

                # Compute FSS for each lead time
                lt_fss = []
                lt_fss_random = []
                for pred_idx in range(preds.shape[0]):
                    pred = preds[pred_idx]

                    obs_evt = (np.abs(truth) >= np.abs(thresh)).astype(float)
                    pred_evt = (np.abs(pred) >= np.abs(thresh)).astype(float)

                    fss_vals = []
                    fssr_vals = []
                    for size in NEIGH_SIZES:
                        fss_vals.append(compute_fss(obs_evt, pred_evt, size))
                        fssr_vals.append(compute_fss_random(obs_evt, size))

                    lt_fss.append(fss_vals)
                    lt_fss_random.append(fssr_vals)

                event_fss.append(lt_fss)
                event_fss_random.append(lt_fss_random)
                n_loaded += 1

            if n_loaded == 0:
                print(f'  [Warning] {model} {param}: no available events')
                continue

            # Convert to arrays
            arr_fss = np.array(event_fss)          # (n_events_loaded, 5, n_sizes)
            arr_fss_random = np.array(event_fss_random)

            all_results[key] = {
                'fss': arr_fss,
                'fss_random': arr_fss_random,
                'n_events': n_loaded,
            }
            print(f'  o {model:12s} {param:20s}  {n_loaded} events')

    print(f'\nTotal {len(all_results)} / {n_models * n_params} (model, param) combinations loaded successfully\n')

    if len(all_results) == 0:
        print('Error: No available data, exiting.')
        return

    # -- Cross-event averaging --
    print('Computing cross-event averages...')

    # Store averaged results
    # mean_fss:   (n_models, n_params, 5, n_sizes)
    # mean_fssr:  (n_models, n_params, 5, n_sizes)
    # std_fss:    (n_models, n_params, 5, n_sizes)
    shape_out = (n_models, n_params, n_leads, n_sizes)
    mean_fss = np.full(shape_out, np.nan)
    mean_fssr = np.full(shape_out, np.nan)
    std_fss = np.full(shape_out, np.nan)
    std_fssr = np.full(shape_out, np.nan)

    for mi, model in enumerate(MODELS):
        for pi, param in enumerate(PARAMETERS):
            key = (model, param)
            if key not in all_results:
                continue

            arr_fss = all_results[key]['fss']           # (n_ev, 5, n_sz)
            arr_fssr = all_results[key]['fss_random']
            n_ev = arr_fss.shape[0]

            for lt in range(n_leads):
                for sz in range(n_sizes):
                    vals_fss = arr_fss[:, lt, sz]
                    vals_fssr = arr_fssr[:, lt, sz]

                    valid_fss = vals_fss[~np.isnan(vals_fss)]
                    valid_fssr = vals_fssr[~np.isnan(vals_fssr)]

                    if len(valid_fss) > 0:
                        mean_fss[mi, pi, lt, sz] = np.mean(valid_fss)
                        std_fss[mi, pi, lt, sz] = np.std(valid_fss, ddof=1) if len(valid_fss) > 1 else 0
                    if len(valid_fssr) > 0:
                        mean_fssr[mi, pi, lt, sz] = np.mean(valid_fssr)
                        std_fssr[mi, pi, lt, sz] = np.std(valid_fssr, ddof=1) if len(valid_fssr) > 1 else 0

    # Save averaged results
    np.savez(str(RESULTS_DIR / 'fss_multiscale_mean.npz'),
             mean_fss=mean_fss,
             mean_fss_random=mean_fssr,
             std_fss=std_fss,
             std_fss_random=std_fssr,
             models=np.array(MODELS, dtype=str),
             parameters=np.array(PARAMETERS, dtype=str),
             lead_times=LEAD_TIMES,
             neighborhood_sizes=NEIGH_SIZES,
             neigh_size_km=np.array([NEIGH_SIZE_KM[s] for s in NEIGH_SIZES]),
             )
    print(f'  Saved: {RESULTS_DIR / "fss_multiscale_mean.npz"}')


    # === Optimal Scale Determination ===
    print('\n' + '=' * 70)
    print('  Optimal scale determination: smallest n where FSS(n) > FSS_random(n)')
    print('=' * 70)

    # For each (model, param, lead_time), find the smallest size such that FSS > FSS_random
    # n_critical: (n_models, n_params, n_leads) stores optimal scale index or NaN
    n_critical_idx = np.full((n_models, n_params, n_leads), np.nan)
    n_critical_size = np.full((n_models, n_params, n_leads), np.nan)

    for mi in range(n_models):
        for pi in range(n_params):
            for lt in range(n_leads):
                fss_vals = mean_fss[mi, pi, lt, :]    # (n_sizes,)
                fssr_vals = mean_fssr[mi, pi, lt, :]

                # Find the first scale where FSS > FSS_random
                found = False
                for sz_idx in range(n_sizes):
                    if (not np.isnan(fss_vals[sz_idx])
                            and not np.isnan(fssr_vals[sz_idx])
                            and fss_vals[sz_idx] > fssr_vals[sz_idx]):
                        n_critical_idx[mi, pi, lt] = sz_idx
                        n_critical_size[mi, pi, lt] = NEIGH_SIZES[sz_idx]
                        found = True
                        break

                if not found:
                    # If FSS never exceeds and is decreasing at the end, all scales fail
                    n_critical_idx[mi, pi, lt] = np.nan
                    n_critical_size[mi, pi, lt] = np.nan

    # -- Print summary table --
    header = f"{'Model':12s} {'Param':20s} {'Lead':>6s} | " + \
             f"{'FSS(3)':>8s} {'FSS(5)':>8s} {'FSS(7)':>8s} {'FSS(11)':>8s} | " + \
             f"{'FSSr(3)':>8s} {'FSSr(5)':>8s} {'FSSr(7)':>8s} {'FSSr(11)':>8s} | " + \
             f"{'BestSz':>7s} {'Best(km)':>9s} {'DeltaFSS':>8s}"
    print()
    print(header)
    print('-' * len(header))

    # Collect all optimal scales for frequency statistics
    all_best_sizes = []
    all_best_km = []

    csv_rows = []

    for mi, model in enumerate(MODELS):
        for pi, param in enumerate(PARAMETERS):
            if np.all(np.isnan(mean_fss[mi, pi, :, :])):
                continue

            for lt in range(n_leads):
                fss_str = []
                fssr_str = []
                for sz in range(n_sizes):
                    v = mean_fss[mi, pi, lt, sz]
                    fss_str.append(f'{v:>8.4f}' if not np.isnan(v) else '     NaN')
                    vr = mean_fssr[mi, pi, lt, sz]
                    fssr_str.append(f'{vr:>8.4f}' if not np.isnan(vr) else '     NaN')

                best_sz = n_critical_size[mi, pi, lt]
                best_sz_idx = n_critical_idx[mi, pi, lt]
                if not np.isnan(best_sz):
                    best_km = NEIGH_SIZE_KM.get(int(best_sz), 0)
                    delta = (mean_fss[mi, pi, lt, int(best_sz_idx)]
                             - mean_fssr[mi, pi, lt, int(best_sz_idx)])
                    all_best_sizes.append(int(best_sz))
                    all_best_km.append(best_km)
                    best_str = f'{int(best_sz):>3d}'
                    best_km_str = f'{best_km:>4d}'
                    delta_str = f'{delta:>8.4f}'
                else:
                    best_str = '   --'
                    best_km_str = '     --'
                    delta_str = '     --'

                print(f"{model:12s} {param:20s} {LEAD_TIMES[lt]:>3d}h | "
                      + ' '.join(fss_str) + ' | ' + ' '.join(fssr_str) + ' | '
                      + f'{best_str:>7s} {best_km_str:>9s} {delta_str:>8s}')

                # CSV row
                csv_rows.append({
                    'model': model,
                    'parameter': param,
                    'lead_time': LEAD_TIMES[lt],
                    'fss_3': mean_fss[mi, pi, lt, 0],
                    'fss_5': mean_fss[mi, pi, lt, 1],
                    'fss_7': mean_fss[mi, pi, lt, 2],
                    'fss_11': mean_fss[mi, pi, lt, 3],
                    'fss_random_3': mean_fssr[mi, pi, lt, 0],
                    'fss_random_5': mean_fssr[mi, pi, lt, 1],
                    'fss_random_7': mean_fssr[mi, pi, lt, 2],
                    'fss_random_11': mean_fssr[mi, pi, lt, 3],
                    'best_size': best_sz if not np.isnan(best_sz) else 0,
                    'best_km': best_km if not np.isnan(best_sz) else 0,
                })

        # Blank line between model groups
        print()

    # Save CSV
    if csv_rows:
        import csv
        csv_path = RESULTS_DIR / 'summary.csv'
        with open(str(csv_path), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f'  Saved: {csv_path}')

    # -- Frequency statistics --
    print('\n' + '=' * 70)
    print('  Scale win frequency')
    print('=' * 70)

    from collections import Counter
    size_counter = Counter(all_best_sizes)
    km_counter = Counter(all_best_km)

    total_decisions = len(all_best_sizes)
    print(f'\n  Total decisions (combinations with skill): {total_decisions}')
    print(f'  Total combinations: {n_models * n_params * n_leads}')
    print()
    print(f'  {"Neighbor":>8s}  {"Physical":>8s}  {"Wins":>8s}  {"Percent":>8s}')
    print(f'  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')
    for sz in sorted(NEIGH_SIZES):
        km = NEIGH_SIZE_KM.get(int(sz), 0)
        count = size_counter.get(int(sz), 0)
        pct = count / total_decisions * 100 if total_decisions > 0 else 0
        bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
        print(f'  {sz:>4d}x{sz:<2d}  {km:>4d} km  {count:>6d}  {pct:>6.1f}%  {bar}')

    print()
    no_skill = n_models * n_params * n_leads - total_decisions
    print(f'  No skill at any scale: {no_skill} / {n_models * n_params * n_leads} '
          f'({no_skill/(n_models*n_params*n_leads)*100:.1f}%)')

    # Save frequency statistics
    csv_count_path = RESULTS_DIR / 'optimal_scale_counts.csv'
    with open(str(csv_count_path), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['neighborhood_size', 'physical_km', 'win_count', 'percentage'])
        for sz in sorted(NEIGH_SIZES):
            km = NEIGH_SIZE_KM.get(int(sz), 0)
            count = size_counter.get(int(sz), 0)
            pct = count / total_decisions * 100 if total_decisions > 0 else 0
            w.writerow([sz, km, count, round(pct, 1)])
        w.writerow(['no_skill', 0, no_skill,
                    no_skill/(n_models*n_params*n_leads)*100])
    print(f'  Saved: {csv_count_path}')


    # === Visualization ===
    print('\n' + '=' * 70)
    print('  Generating scale curves...')
    print('=' * 70)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # y-axis labels: physical km of neighborhood scales
        size_labels_fig = [f'{s}x{s}\n({NEIGH_SIZE_KM.get(int(s), 0)} km)' for s in NEIGH_SIZES]
        x_pos = np.arange(n_sizes)  # 0, 1, 2, 3

        # -- Figure 1: One figure per parameter, 4 models x 5 lead times --
        for pi, param in enumerate(PARAMETERS):
            has_data = False
            for mi in range(n_models):
                if not np.all(np.isnan(mean_fss[mi, pi, :, :])):
                    has_data = True
                    break
            if not has_data:
                continue

            fig, axes = plt.subplots(n_models, n_leads, figsize=(4 * n_leads, 3.5 * n_models))
            if n_models == 1:
                axes = axes[np.newaxis, :]
            if n_leads == 1:
                axes = axes[:, np.newaxis]

            for mi, model in enumerate(MODELS):
                for lt in range(n_leads):
                    ax = axes[mi, lt]

                    fss = mean_fss[mi, pi, lt, :]
                    fss_std = std_fss[mi, pi, lt, :]
                    fssr = mean_fssr[mi, pi, lt, :]

                    # FSS(n) curve
                    valid = ~np.isnan(fss)
                    if valid.any():
                        ax.errorbar(x_pos[valid], fss[valid],
                                    yerr=fss_std[valid] if valid.any() else None,
                                    color='#1a76b5', linewidth=2.2, marker='o', markersize=6,
                                    label='FSS(n)', capsize=3)

                    # FSS_random(n) curve
                    valid_r = ~np.isnan(fssr)
                    if valid_r.any():
                        ax.plot(x_pos[valid_r], fssr[valid_r], '--',
                                color='#d62728', linewidth=2.0, marker='s', markersize=5,
                                label='FSS$_{\\mathrm{random}}$(n)')

                    # Annotate optimal scale
                    best_sz_idx = n_critical_idx[mi, pi, lt]
                    if not np.isnan(best_sz_idx):
                        best_idx_int = int(best_sz_idx)
                        best_val = mean_fss[mi, pi, lt, best_idx_int]
                        ax.annotate(f'<- best\n({int(NEIGH_SIZES[best_idx_int])}x{int(NEIGH_SIZES[best_idx_int].item())})',
                                    xy=(x_pos[best_idx_int], best_val),
                                    xytext=(x_pos[best_idx_int] + 0.3, best_val - 0.08),
                                    fontsize=7, color='darkgreen', fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2))

                    # No skill annotation
                    if np.isnan(best_sz_idx):
                        ax.text(0.5, 0.5, 'No skill\nat any scale',
                                transform=ax.transAxes, ha='center', va='center',
                                fontsize=9, color='gray', fontstyle='italic',
                                alpha=0.7)

                    # Annotations
                    if mi == 0:
                        ax.set_title(f'{LEAD_TIMES[lt]:.0f}h', fontsize=12, fontweight='bold')
                    if lt == 0:
                        ax.set_ylabel(f'{MODEL_LABELS[model]}', fontsize=11, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(size_labels_fig, fontsize=7)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

                    # Show legend only in bottom right corner
                    if mi == n_models - 1 and lt == n_leads - 1:
                        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

            fig.suptitle(f'{PARAM_LABELS.get(param, param)} -- '
                         f'FSS(n) vs FSS_random(n)  (mean over {n_events} events)',
                         fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            save_path = PLOTS_DIR / f'scale_curve_{param}.png'
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  o {save_path.name}')

        # -- Figure 2: Optimal scale heatmap --
        # For each lead_time, show models x parameters matrix
        for lt in range(n_leads):
            fig, ax = plt.subplots(figsize=(10, 5))

            # Extract optimal scale (as size value, not index)
            heat_data = n_critical_size[:, :, lt]  # (n_models, n_params)
            heat_data_display = np.copy(heat_data)

            # Color mapping: no skill = 0, others mapped to 1-4
            cmap_data = np.zeros_like(heat_data)
            for mi in range(n_models):
                for pi in range(n_params):
                    if not np.isnan(heat_data[mi, pi]):
                        sz = int(heat_data[mi, pi])
                        sz_idx = np.where(NEIGH_SIZES == sz)[0][0]
                        cmap_data[mi, pi] = sz_idx + 1  # 1-indexed: 1,2,3,4
                    else:
                        cmap_data[mi, pi] = 0  # no skill

            # Custom colormap
            from matplotlib.colors import ListedColormap, BoundaryNorm
            n_colors = len(NEIGH_SIZES) + 1  # No skill + 4 scales
            cmap = ListedColormap(['#f0f0f0', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'][:n_colors])
            bounds = np.arange(-0.5, n_colors + 0.5, 1)
            norm = BoundaryNorm(bounds, cmap.N, extend='neither')

            im = ax.imshow(cmap_data, aspect='auto', cmap=cmap, norm=norm)

            # Text annotations
            for mi in range(n_models):
                for pi in range(n_params):
                    if not np.isnan(heat_data[mi, pi]):
                        sz = int(heat_data[mi, pi])
                        km = NEIGH_SIZE_KM.get(sz, 0)
                        text = f'{sz}x{sz}\n{km}km'
                        ax.text(pi, mi, text, ha='center', va='center',
                                fontsize=8, fontweight='bold', color='black')
                    else:
                        ax.text(pi, mi, 'x', ha='center', va='center',
                                fontsize=10, color='gray')

            ax.set_xticks(range(n_params))
            ax.set_xticklabels([PARAM_LABELS.get(p, p) for p in PARAMETERS], rotation=30, ha='right', fontsize=10)
            ax.set_yticks(range(n_models))
            ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
            ax.set_title(f'Optimal Neighborhood Scale -- {LEAD_TIMES[lt]:.0f}h forecast', fontsize=13, fontweight='bold')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=0.6)
            cbar.set_ticklabels(['No skill'] + [f'{s}x{s} ({NEIGH_SIZE_KM.get(int(s), 0)}km)'
                                                 for s in NEIGH_SIZES], fontsize=8)

            plt.tight_layout()
            save_path = PLOTS_DIR / f'heatmap_optimal_scale_{LEAD_TIMES[lt]:.0f}h.png'
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  o {save_path.name}')

        # -- Figure 3: Summary bar chart (optimal scale frequency distribution) --
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes_display = [f'{sz}x{sz}\n({NEIGH_SIZE_KM.get(int(sz), 0)} km)' for sz in sorted(NEIGH_SIZES)]
        counts = [size_counter.get(int(sz), 0) for sz in sorted(NEIGH_SIZES)]
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

        bars = ax.bar(range(len(sizes_display)), counts, color=colors, width=0.6, edgecolor='white')
        for bar, count in zip(bars, counts):
            pct = count / total_decisions * 100 if total_decisions > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(sizes_display)))
        ax.set_xticklabels(sizes_display, fontsize=10)
        ax.set_ylabel('Win count', fontsize=12)
        ax.set_title(f'Optimal Neighborhood Scale Frequency\n(Total {total_decisions} decisions, {n_models * n_params * n_leads} combinations)',
                     fontsize=13, fontweight='bold')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # Annotate no skill
        if no_skill > 0:
            ax.text(0.98, 0.95, f'No skill at any scale: {no_skill}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, color='gray', fontstyle='italic')

        plt.tight_layout()
        save_path = PLOTS_DIR / 'optimal_scale_distribution.png'
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  o {save_path.name}')

        print('\n  All plots saved to:', PLOTS_DIR)

    except ImportError:
        print('  [Warning] matplotlib not installed, skipping plots.')


    # === Final Recommendation ===
    print('\n' + '=' * 70)
    print('  Final Recommendation')
    print('=' * 70)

    if total_decisions > 0:
        winning_size = size_counter.most_common(1)[0][0]
        winning_km = NEIGH_SIZE_KM.get(winning_size, 0)
        winning_pct = size_counter[winning_size] / total_decisions * 100

        print(f'\n  Among {total_decisions} decisions with skill:')
        print(f'  Most frequently selected optimal neighborhood:  {winning_size}x{winning_size}  (~{winning_km} km)')
        print(f'  Percentage:                 {winning_pct:.1f}%')
        print()

        # Weighted average suggestion
        weighted_avg_km = sum(NEIGH_SIZE_KM.get(s, s * 25) * c
                              for s, c in size_counter.items()) / total_decisions
        print(f'  Weighted average optimal scale:     ~{weighted_avg_km:.0f} km')
        print()

        # Detailed statistics per scale
        print(f'  Complete frequency:')
        for sz in sorted(NEIGH_SIZES):
            km = NEIGH_SIZE_KM.get(int(sz), 0)
            count = size_counter.get(int(sz), 0)
            pct = count / total_decisions * 100
            bar = '#' * int(pct) + '.' * (50 - int(pct))
            print(f'    {sz}x{sz} ({km:>3d} km):  {count:>3d} times ({pct:>5.1f}%)  {bar}')

        print(f'\n  Recommendation: use {winning_size}x{winning_size} (~{winning_km} km)'
              ' as the FSS neighborhood scale')
        if no_skill > 0:
            print(f'  Note: {no_skill}/{n_models * n_params * n_leads} combinations have no skill'
                  ' at any scale')
    else:
        print('\n  All combinations have no skill at any scale.')
        print('  Consider checking threshold settings or data quality.')

    print('\n' + '=' * 70)
    print('  Analysis complete! Results directory:')
    print(f'    CSV data:  {RESULTS_DIR}')
    print(f'    Scale curves: {PLOTS_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
