import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cartopy import crs as ccrs, feature as cfeature
import os


def plot_all_models(time, parameter, lead_idx):
    """
    Plot 6 fields for a given time, parameter, and lead time:
      Real, GFS[0] (GFS analysis field), Pangu, FengWu v1, FengWu v2, FuXi
    All subplots overlaid with a red contour (Real's 75th percentile) for comparison.
    """
    lead_times = [24, 48, 72, 120, 168]
    lead_label = lead_times[lead_idx]
    models_ai = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
    model_labels = {
        'pangu': 'Pangu',
        'fengwu_v1': 'FengWu v1',
        'fengwu_v2': 'FengWu v2',
        'fuxi': 'FuXi',
    }

    # ---------- Load data ----------
    real_path = os.path.join(time, f'{parameter}_real.npy')
    if not os.path.exists(real_path):
        print(f"Real data not found: {real_path}")
        return
    real_data = np.load(real_path)

    gfs_path = os.path.join(time, 'gfs', f'{parameter}_gfs.npy')
    if os.path.exists(gfs_path):
        gfs_all = np.load(gfs_path)
        gfs_analysis = gfs_all[0]          # GFS analysis field
        gfs_fcst = gfs_all[1:][lead_idx]   # GFS forecast (same lead_idx)
    else:
        print(f"GFS data not found: {gfs_path}")
        gfs_analysis = None
        gfs_fcst = None

    # AI models
    model_data = {}
    for m in models_ai:
        p = os.path.join(time, m, f'{parameter}_{m}.npy')
        if os.path.exists(p):
            arr = np.load(p)
            model_data[m] = arr[lead_idx]  # (21, 25)
        else:
            print(f"Data not found: {p}")

    # ---------- Compute Real's P75 (unified red reference contour) ----------
    p75 = np.nanpercentile(real_data, 75)
    print(f"Real P75 = {p75:.2f}")

    # ---------- Map parameters ----------
    projPC = ccrs.PlateCarree()
    lonW, lonE = -100.0, -94.0
    latS, latN = 33.0, 38.0
    lat = np.linspace(latN, latS, 21)
    lon = np.linspace(lonW % 360, lonE % 360, 25)
    lonall, latall = np.meshgrid(lon, lat)

    # ---------- Unified color scale ----------
    all_vals = [real_data.ravel()]
    if gfs_analysis is not None:
        all_vals.append(gfs_analysis.ravel())
    if gfs_fcst is not None:
        all_vals.append(gfs_fcst.ravel())
    for m in models_ai:
        if m in model_data:
            all_vals.append(model_data[m].ravel())
    combined = np.concatenate(all_vals)
    combined = combined[~np.isnan(combined)]
    if len(combined) == 0:
        combined = np.array([0, 1])
    vmin, vmax = np.percentile(combined, [1, 99])
    levels = np.linspace(vmin, vmax, 12)

    # ---------- Subplot layouts (horizontal + vertical) ----------
    titles_def = [
        ('ERA5-Reanalysis', real_data),
        ('GFS fct0', gfs_analysis),
        ('GFS Forecast', gfs_fcst),
        ('Pangu', model_data.get('pangu')),
        ('FengWu v1', model_data.get('fengwu_v1')),
        ('FengWu v2', model_data.get('fengwu_v2')),
        ('FuXi', model_data.get('fuxi')),
    ]

    layout_configs = [
        {'tag': 'horizontal', 'figsize': (28, 10)},
        {'tag': 'vertical',   'figsize': (14, 22)},
    ]

    for cfg in layout_configs:
        fig = plt.figure(figsize=cfg['figsize'])
        tag = cfg['tag']

        if tag == 'horizontal':
            # 2 rows, row 0 has 5 columns (middle 3 for plots, left/right white space)
            gs = gridspec.GridSpec(
                2, 1, figure=fig, height_ratios=[1, 1], hspace=0.10)
            gs0 = gridspec.GridSpecFromSubplotSpec(
                1, 5, gs[0], width_ratios=[0.4, 1, 1, 1, 0.4], wspace=0.06)
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1], wspace=0.06)
        else:
            nrows_v, ncols_v = 4, 2

        for idx, (title, data) in enumerate(titles_def):
            if data is None:
                continue

            if tag == 'horizontal':
                if idx < 3:
                    ax = fig.add_subplot(gs0[idx + 1], projection=projPC)
                else:
                    ax = fig.add_subplot(gs1[idx - 3], projection=projPC)
            else:
                ax = fig.add_subplot(nrows_v, ncols_v, idx + 1, projection=projPC)

            ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
            ax.add_feature(cfeature.STATES)
            gl = ax.gridlines(draw_labels=True, dms=True, color='black',
                              linestyle='--', alpha=0.4)
            gl.xlines = False
            gl.ylines = False
            gl.top_labels = False
            gl.right_labels = False
            ax.coastlines(resolution='110m', linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

            # Filled contours
            cf = ax.contourf(lonall, latall, data, levels=levels,
                             transform=projPC, cmap='coolwarm',
                             extend='both', alpha=0.8)

            # Red contour — Real's P75
            c_ref = ax.contour(lonall, latall, data, levels=[p75],
                               transform=projPC, colors='red', linewidths=2)
            if idx == 0:
                plt.clabel(c_ref, inline=True, fontsize=10, colors='red',
                           fmt=f'P75={p75:.0f}')

            # Black contours
            c = ax.contour(lonall, latall, data, levels=levels,
                           transform=projPC, colors='k', linewidths=0.6, alpha=0.35)
            plt.clabel(c, inline=True, fontsize=6, colors='black')

            # Title
            if title in ('ERA5-Reanalysis', 'GFS fct0'):
                subtitle = title
            else:
                subtitle = f'{title} {lead_label}h'
            ax.set_title(subtitle, fontsize=13, fontweight='bold')
            cb = plt.colorbar(cf, ax=ax, shrink=0.5, pad=0.04, format='%d')
            # Only show integer ticks, avoid multiple levels mapping to the same integer
            tick_min = int(np.ceil(vmin))
            tick_max = int(np.floor(vmax))
            if tick_max > tick_min:
                n_int = tick_max - tick_min
                if n_int <= 8:
                    cb.set_ticks(np.arange(tick_min, tick_max + 1))
                else:
                    step = max(1, n_int // 8)
                    cb.set_ticks(np.arange(tick_min, tick_max + 1, step))

        # Hide the 8th empty spot in vertical layout
        if tag == 'vertical':
            ax_empty = fig.add_subplot(nrows_v, ncols_v, 8)
            ax_empty.axis('off')

        plt.suptitle(
            f'{parameter.upper()} — {time}\n'
            f'Red contour: P75 of observed data ({p75:.1f})',
            fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out_dir = os.path.join('single_plots', time)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(
            out_dir, f'{parameter}_lead{lead_label}h_{tag}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)


# ==================== Usage Example ====================
if __name__ == '__main__':
    plot_all_models(
        time='2025-05-19-23UTC',
        parameter='stp',
        lead_idx=0,     # 24h
    )
