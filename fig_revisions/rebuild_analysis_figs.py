"""Rebuild Figure 10 (multi-model vertical profiles). Run with conda base."""
import os
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Volumes/T7/Research/spatial_evaluation'
OUT = '/Volumes/T7/Research/spatial_evaluation/fig_revisions/output'
os.makedirs(OUT, exist_ok=True)

CASES = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC',
         '2023-04-20-02UTC', '2024-04-28-04UTC', '2024-05-07-02UTC',
         '2024-05-26-04UTC', '2025-05-19-23UTC']

MODELS = [('pangu', 'Pangu', '#1f77b4', 'o'),
          ('fengwu_v2', 'FengWu v2', '#42949E', '^'),
          ('fengwu_v1', 'FengWu v1', '#8BCF8B', 's'),
          ('fuxi', 'FuXi', '#d62728', 'D')]

PRESS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

LAT_MIN, LAT_MAX = 33.0, 38.0
LON_MIN, LON_MAX = 260.0, 266.0


# 全球 0.25° 网格索引: lat 90->-90, lon 0->360
LI0, LI1 = int(round((90.0 - LAT_MAX) / 0.25)), int(round((90.0 - LAT_MIN) / 0.25))
LJ0, LJ1 = int(round(LON_MIN / 0.25)), int(round(LON_MAX / 0.25))


def _crop_era(ds_e, p):
    """ERA5 指定层 (lat, lon) 切片 (21, 25)."""
    li = PRESS.index(p)
    t = ds_e['t'].values[0, li, LI0:LI1, LJ0:LJ1]
    q = ds_e['q'].values[0, li, LI0:LI1, LJ0:LJ1]
    return t - 273.15, q * 1000


def _crop_model(ds_f, model, p, var):
    """模型指定层 (lat, lon) 索引切片. 返回 C / g/kg."""
    if model == 'pangu':
        li = PRESS.index(p)
        nm = {'T': 'temperature', 'q': 'specific_humidity'}[var]
        arr = ds_f[nm].values[li, LI0:LI1, LJ0:LJ1]
    elif model == 'fuxi':
        if var == 'T':
            arr = ds_f[f'T{p}'].values[LI0:LI1, LJ0:LJ1]
        else:
            return None
    else:
        nm = f't{p}' if var == 'T' else f'q{p}'
        if nm not in ds_f.data_vars:
            return None
        arr = ds_f[nm].values[LI0:LI1, LJ0:LJ1]
    if var == 'T':
        return arr - 273.15
    return arr * 1000


def profile_bias(case, model):
    ds_e = xr.open_dataset(os.path.join(BASE, case, 'real.nc'))
    if model == 'fuxi':
        f_path = os.path.join(BASE, case, 'fuxi', 'fuxi_24.nc')
    else:
        f_path = os.path.join(BASE, case, model, f'{model}_24.nc')
    ds_f = xr.open_dataset(f_path)
    out = {}
    for p in PRESS:
        try:
            e_t, e_q = _crop_era(ds_e, p)
            f_t = _crop_model(ds_f, model, p, 'T')
            if f_t is None:
                continue
            e_td = mpcalc.dewpoint_from_specific_humidity(
                p * units.hPa, e_q * units('g/kg')).magnitude
            if model == 'fuxi':
                r_name = f'R{p}'
                if r_name not in ds_f.data_vars:
                    continue
                f_td = mpcalc.dewpoint_from_relative_humidity(
                    f_t * units.degC, ds_f[r_name].values[LI0:LI1, LJ0:LJ1] * units.percent).magnitude
            else:
                f_q = _crop_model(ds_f, model, p, 'q')
                if f_q is None:
                    continue
                f_td = mpcalc.dewpoint_from_specific_humidity(
                    p * units.hPa, (f_q) * units('g/kg')).magnitude
            # 统一尺寸 (可能 1 像素差异)
            nlat, nlon = e_t.shape
            f_t = f_t[:nlat, :nlon]
            f_td = f_td[:nlat, :nlon]
            # θe (Bolton 1980 近似)
            th_e = mpcalc.equivalent_potential_temperature(
                p * units.hPa, e_t * units.degC, e_td * units.degC).magnitude
            th_f = mpcalc.equivalent_potential_temperature(
                p * units.hPa, f_t * units.degC, f_td * units.degC).magnitude
            step = 3
            sel = (slice(None, None, step), slice(None, None, step))
            dT = (f_t[sel] - e_t[sel]).ravel()
            dTd = (f_td[sel] - e_td[sel]).ravel()
            dTe = (th_f[sel] - th_e[sel]).ravel()
            dT = dT[np.isfinite(dT)]
            dTd = dTd[np.isfinite(dTd)]
            dTe = dTe[np.isfinite(dTe)]
            if len(dT) > 0 and len(dTd) > 0 and len(dTe) > 0:
                out[p] = (np.mean(dT), np.mean(dTd), np.mean(dTe))
        except Exception as e:
            print(f'      {p} hPa: {e}')
            continue
    ds_e.close(); ds_f.close()
    return out


def fig10_profiles():
    print('Figure 10: multi-model profiles (10 cases, 4 models)...')
    prof = {}
    for mname, *_ in MODELS:
        prof[mname] = {'T': {}, 'Td': {}, 'theta_e': {}}
    for ci, case in enumerate(CASES):
        print(f'  [{ci+1}/10] {case[:10]}...')
        for mname, *_ in MODELS:
            try:
                bias = profile_bias(case, mname)
            except Exception as e:
                print(f'    {mname} error: {e}')
                continue
            for p, vals in bias.items():
                dT, dTd, dTe = vals
                prof[mname]['T'].setdefault(p, []).append(dT)
                prof[mname]['Td'].setdefault(p, []).append(dTd)
                prof[mname]['theta_e'].setdefault(p, []).append(dTe)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, kind, title, xlab in [
            (axes[0], 'T', 'Temperature bias', '$\\Delta T$ ($^\\circ$C)'),
            (axes[1], 'Td', 'Dew point temperature bias', '$\\Delta T_d$ ($^\\circ$C)'),
            (axes[2], 'theta_e', '$\\theta_e$ bias', '$\\Delta\\theta_e$ (K)')]:
        for mname, label, color, marker in MODELS:
            ps = sorted(prof[mname][kind].keys())
            if not ps:
                continue
            vals = [np.mean(prof[mname][kind][p]) for p in ps]
            ax.plot(vals, ps, marker=marker, color=color, label=label,
                    linewidth=2, markersize=9)
        ax.set_ylim(1000, 500)
        ax.set_xlabel(xlab, fontsize=18, fontweight='bold')
        ax.set_ylabel('Pressure (hPa)', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=15)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=14, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_multi_model_profiles.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig_multi_model_profiles.png')


if __name__ == '__main__':
    fig10_profiles()
