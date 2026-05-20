import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import os


def plot_real_forecast(real_data, forecast_data, param_name, time_str,
                       lead_idx=None, model_name='forecast'):
    """
    Plot real (left) and forecast (right) side by side, with a red contour
    at the 75th percentile of the observed data.
    """
    # Compute the 75th percentile of the observed data as a red reference line
    p75 = np.nanpercentile(real_data, 75)
    print(f"75th percentile of observed data: {p75:.2f}")

    # Map setup
    projPC = ccrs.PlateCarree()
    lonW, lonE = -100.0, -94.0
    latS, latN = 33.0, 38.0

    lat = np.linspace(latN, latS, 21)
    lon = np.linspace(lonW % 360, lonE % 360, 25)
    lonall, latall = np.meshgrid(lon, lat)

    # Unified color scale range
    combined = np.concatenate([real_data.ravel(), forecast_data.ravel()])
    combined = combined[~np.isnan(combined)]
    if len(combined) == 0:
        combined = np.array([0, 1])
    vmin, vmax = np.percentile(combined, [1, 99])
    levels = np.linspace(vmin, vmax, 20)

    # Title suffix
    lead_suffix = f' - Lead {lead_idx}h' if lead_idx is not None else ''

    fig = plt.figure(figsize=(22, 8.5))

    for col_idx, (data, subtitle) in enumerate([
        (real_data, f'Observed ({time_str})'),
        (forecast_data, f'{model_name} ({time_str}){lead_suffix}')
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection=projPC)
        ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
        ax.add_feature(cfeature.STATES)
        gl = ax.gridlines(draw_labels=True, dms=True, color='black',
                          linestyle='--', alpha=0.5)
        gl.xlines = False
        gl.ylines = False
        gl.top_labels = False
        gl.right_labels = False
        ax.coastlines(resolution='110m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

        # Filled contours
        cf = ax.contourf(lonall, latall, data, levels=levels,
                         transform=projPC, cmap='viridis',
                         extend='both', alpha=0.8)

        # Red contour: 75th percentile of observed data (drawn on both sides for comparison)
        c_ref = ax.contour(lonall, latall, data, levels=[p75],
                           transform=projPC, colors='red', linewidths=2)
        plt.clabel(c_ref, inline=True, fontsize=10, colors='red',
                   fmt=f'P75={p75:.0f}')

        # Black contours
        c = ax.contour(lonall, latall, data, levels=levels,
                       transform=projPC, colors='k', linewidths=0.6, alpha=0.4)
        plt.clabel(c, inline=True, fontsize=7, colors='black')

        ax.set_title(subtitle, fontsize=14, fontweight='bold')
        plt.colorbar(cf, ax=ax, shrink=0.6, pad=0.05)

    plt.suptitle(f'{param_name.upper()} — Red contour at P75 of observed data',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


# ==================== Usage Example ====================
if __name__ == '__main__':
    time = '2024-04-28-04UTC'
    parameter = 'lcl'
    model = 'fengwu_v1'
    lead_idx = 1  # Corresponds to 168h lead time

    # Load real data
    real_path = os.path.join(time, f'{parameter}_real.npy')
    # Load forecast data
    fcst_path = os.path.join(time, model, f'{parameter}_{model}.npy')

    if os.path.exists(real_path) and os.path.exists(fcst_path):
        real_data = np.load(real_path)
        fcst_data = np.load(fcst_path)

        # forecast is (5, 21, 25), take the slice corresponding to lead_idx
        if fcst_data.ndim == 3:
            fcst_slice = fcst_data[lead_idx]
        else:
            fcst_slice = fcst_data

        plot_real_forecast(
            real_data, fcst_slice,
            param_name=parameter,
            time_str=time,
            lead_idx=[24, 48, 72, 120, 168][lead_idx],
            model_name=model)
    else:
        print(f"File not found, please check paths.")
        print(f"  Real: {real_path}")
        print(f"  Forecast: {fcst_path}")
