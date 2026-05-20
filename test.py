import os
import numpy as np
import xarray as xr

INPUT_DIR = "/Volumes/T7/Research/FuXi-Sky/inputs"

nc_files = sorted(os.listdir(INPUT_DIR))

for fname in nc_files:
    path = os.path.join(INPUT_DIR, fname)
    data = xr.open_dataarray(path)
    arr = data.values  # (time=2, level=70, lat=721, lon=1440)

    has_nan = np.isnan(arr).any()
    if not has_nan:
        continue

    print(f"\n{'='*60}")
    print(f"❌ {fname}  shape={arr.shape}")
    print(f"{'='*60}")

    # 逐 time × level 检查 NaN
    for t in range(2):
        for lev in range(70):
            layer = arr[t, lev]
            nan_count = np.isnan(layer).sum()
            if nan_count > 0:
                total = layer.size
                pct = nan_count / total * 100
                print(f"   time={t}, level={lev:2d} ({data.level.values[lev]}): "
                      f"NaN {nan_count}/{total} ({pct:.1f}%)")
