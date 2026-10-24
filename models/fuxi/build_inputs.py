"""
Build model input files for all (target, lead) combinations.
Each input.nc has shape (time=2, level=70, lat=721, lon=1440).
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# Constants
# ============================================================
pl_names = ['z', 't', 'u', 'v', 'r']
sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
levels = [50, 100, 150, 200, 250, 300, 400, 500,
          600, 700, 850, 925, 1000]

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

TARGET_TIMES = ["2023-04-20 02:00"]
LEAD_HOURS = [24, 48, 72, 120, 168]


# ============================================================
# ERA5 Snapshot Reader (from make_era5_input.ipynb)
# ============================================================
def make_era5_snapshot(dt, data_dir):
    """Read ERA5 at a given datetime, return 70-channel DataArray."""
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")
    folder = os.path.join(data_dir, f"ERA5{date_str}")

    print(f"  Reading: {dt}  ({folder})")

    # --- Pressure level ---
    p_path = os.path.join(folder, f"P{date_str}{hour_str}.nc")
    pl = xr.open_dataset(p_path)
    if 'valid_time' in pl.dims:
        pl = pl.rename({'valid_time': 'time'})
    if 'expver' in pl.dims:
        pl = pl.isel(expver=0)
    if 'number' in pl.dims:
        pl = pl.isel(number=0)

    # --- Surface ---
    s_path = os.path.join(folder, f"S{date_str}{hour_str}.nc")
    sfc = xr.open_dataset(s_path)
    if 'valid_time' in sfc.dims:
        sfc = sfc.rename({'valid_time': 'time'})
    if 'expver' in sfc.dims:
        sfc = sfc.isel(expver=0)
    if 'number' in sfc.dims:
        sfc = sfc.isel(number=0)
    # 强制 time 坐标（某些下载的 S 文件内部 time 与实际时刻不符）
    sfc['time'] = [dt]

    # --- Daily precip (6h rolling sum, m → mm) ---
    # Load hourly R data; concat with previous day for cross-boundary windows
    r_path = os.path.join(folder, f"R{date_str}.nc")
    tp = xr.open_dataarray(r_path)
    if 'valid_time' in tp.dims:
        tp = tp.rename({'valid_time': 'time'})

    prev_date = dt - pd.Timedelta(days=1)
    prev_r_path = os.path.join(
        data_dir, f"ERA5{prev_date.strftime('%Y%m%d')}",
        f"R{prev_date.strftime('%Y%m%d')}.nc")
    if os.path.exists(prev_r_path):
        tp_prev = xr.open_dataarray(prev_r_path)
        if 'valid_time' in tp_prev.dims:
            tp_prev = tp_prev.rename({'valid_time': 'time'})
        tp = xr.concat([tp_prev, tp], dim='time')

    tp = tp.rolling(time=6).sum() * 1000
    tp = tp.sel(time=dt)
    tp = tp.clip(min=0, max=1000)
    tp = tp.expand_dims(time=[dt])
    sfc['tp'] = tp

    # --- Assemble 70 channels ---
    ds_list = []
    for name in pl_names + sfc_names:
        if name in pl_names:
            v = pl[name]
            if 'pressure_level' in v.dims:
                v = v.sel(pressure_level=levels)
                v = v.rename({'pressure_level': 'level'})
        else:
            v = sfc[name]
            v = v.expand_dims(level=[1])

        if 'valid_time' in v.dims:
            v = v.rename({'valid_time': 'time'})
        v.name = name
        v.attrs = {}
        print(f"    {name}: {v.shape}, {float(v.min()):.1f} ~ {float(v.max()):.1f}")
        ds_list.append(v)

    ds = xr.concat(ds_list, dim='level')

    # Channel names + coordinate standardisation
    channel_names = [f'{n.upper()}{l}' for n in pl_names for l in levels]
    channel_names += [n.upper() for n in sfc_names]
    ds = ds.assign_coords(level=channel_names[:len(ds.level)])
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    ds = ds.sortby('lat')
    ds = ds.astype(np.float32)
    ds = ds.isel(time=0)

    return ds


def build_input(init_time, data_root):
    """Build model input (time=2, level=70, lat=721, lon=1440)."""
    init_time = pd.to_datetime(init_time)
    t_minus_6 = init_time - pd.Timedelta(hours=6)

    ds_t6 = make_era5_snapshot(t_minus_6, data_root)
    ds_t = make_era5_snapshot(init_time, data_root)

    ds_input = xr.concat([ds_t6, ds_t], dim='time')

    if ds_input.lat.values[0] < ds_input.lat.values[-1]:
        ds_input = ds_input.sortby("lat", ascending=False)

    assert ds_input.lat.values[-1] == -90, f"lat[-1]={ds_input.lat.values[-1]} != -90"
    assert ds_input.shape == (2, 70, 721, 1440), f"shape={ds_input.shape}"

    return ds_input


# ============================================================
# Main
# ============================================================
total = len(TARGET_TIMES) * len(LEAD_HOURS)
count = 0

for target_str in TARGET_TIMES:
    target_time = pd.to_datetime(target_str)
    target_label = target_time.strftime("%Y%m%d_%H")

    for lead in LEAD_HOURS:
        count += 1
        init_time = target_time - pd.Timedelta(hours=lead)
        label = f"{target_label}_lead{lead:03d}h"

        print(f"\n{'='*60}")
        print(f"[{count}/{total}] Target: {target_time}  Lead: {lead}h")
        print(f"  Init time: {init_time}")
        print(f"{'='*60}")

        data = build_input(init_time, DATA_ROOT)

        # Save to input directory
        input_dir = os.path.join(DATA_ROOT, "inputs")
        os.makedirs(input_dir, exist_ok=True)
        out_path = os.path.join(input_dir, f"input_{label}.nc")
        data.to_netcdf(out_path)

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✅ Saved: {out_path}")
        print(f"     Shape: {data.shape}, Size: {size_mb:.1f} MB")
        print(f"     Lat range: {float(data.lat.values[0]):.2f} → {float(data.lat.values[-1]):.2f}")
        print(f"     Levels: {len(data.level)}")

print(f"\n{'='*60}")
print(f"All {total} input files generated in {os.path.join(DATA_ROOT, 'inputs')}/")
