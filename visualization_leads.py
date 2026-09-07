import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
from matplotlib.lines import Line2D
import os


def plot_model_leads(time, parameter, model='pangu'):
    """
    Plot all forecast lead times + Real field for a given time, parameter, and model.
    Default layout 2x3, each subplot overlaid with the Real 75th percentile red contour.
    """
    lead_times = [24, 48, 72, 120, 168]
    model_labels = {
        'pangu': 'Pangu', 'fengwu_v1': 'FengWu v1',
        'fengwu_v2': 'FengWu v2', 'fuxi': 'FuXi', 'gfs': 'GFS',
    }
    model_label = model_labels.get(model, model)

    # ---------- Load data ----------
    real_path = os.path.join(time, f'{parameter}_real.npy')
    if not os.path.exists(real_path):
        print(f"Real data not found: {real_path}")
        return
    real_data = np.load(real_path)

    # Load model forecast (5, 21, 25)
    if model == 'gfs':
        fcst_path = os.path.join(time, 'gfs', f'{parameter}_gfs.npy')
        if not os.path.exists(fcst_path):
            print(f"GFS data not found: {fcst_path}")
            return
        fcst_all = np.load(fcst_path)
        if fcst_all.ndim == 3 and fcst_all.shape[0] == 6:
            fcst_all = fcst_all[1:]  # Remove analysis field, keep 5 lead times
    else:
        fcst_path = os.path.join(time, model, f'{parameter}_{model}.npy')
        if not os.path.exists(fcst_path):
            print(f"Data not found: {fcst_path}")
            return
        fcst_all = np.load(fcst_path)  # (5, 21, 25)

    # ---------- Compute Real's P75 ----------
    p75 = np.nanpercentile(real_data, 75)
    print(f"Real P75 = {p75:.2f} J/kg")

    # ---------- Map parameters ----------
    projPC = ccrs.PlateCarree()
    lonW, lonE = -100.0, -94.0
    latS, latN = 33.0, 38.0
    lat = np.linspace(latN, latS, 21)
    lon = np.linspace(lonW % 360, lonE % 360, 25)
    lonall, latall = np.meshgrid(lon, lat)

    # ---------- Unified color scale ----------
    all_vals = [real_data.ravel()]
    for i in range(5):
        all_vals.append(fcst_all[i].ravel())
    combined = np.concatenate(all_vals)
    combined = combined[~np.isnan(combined)]
    if len(combined) == 0:
        combined = np.array([0, 1])
    vmin, vmax = np.percentile(combined, [1, 99])
    levels = np.linspace(vmin, vmax, 20)

    # ---------- Subplot layout 2x3 ----------
    titles_def = [
        ('ERA5-Reanalysis', real_data),
        (f'{model_label} {lead_times[0]}h', fcst_all[0]),
        (f'{model_label} {lead_times[1]}h', fcst_all[1]),
        (f'{model_label} {lead_times[2]}h', fcst_all[2]),
        (f'{model_label} {lead_times[3]}h', fcst_all[3]),
        (f'{model_label} {lead_times[4]}h', fcst_all[4]),
    ]

    fig = plt.figure(figsize=(20, 14))

    for idx, (title, data) in enumerate(titles_def):
        ax = fig.add_subplot(2, 3, idx + 1, projection=projPC)
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

        cf = ax.contourf(lonall, latall, data, levels=levels,
                         transform=projPC, cmap='viridis',
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

        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.colorbar(cf, ax=ax, shrink=0.5, pad=0.04)

    plt.suptitle(
        f'{parameter.upper()} — {time}\n'
        f'Model: {model_label} — Red contour: P75 of ERA5 ({p75:.1f})',
        fontsize=16, fontweight='bold', y=0.97)

    # Legend — parameter label only
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2,
               label=f'ERA5 P75 ({p75:.0f})'),
        Line2D([0], [0], color='black', linewidth=0.6, alpha=0.35,
               label=parameter.upper()),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save
    out_dir = os.path.join('single_plots', time)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(
        out_dir, f'{parameter}_{model}_all_leads.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ==================== Usage Example ====================
if __name__ == '__main__':
    plot_model_leads(
        time='2025-05-19-23UTC',
        parameter='positive_helicity',
        model='gfs',       # Options: pangu, fengwu_v1, fengwu_v2, fuxi, gfs
    )
