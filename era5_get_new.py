import os
import zipfile
import shutil
import tempfile

import cdsapi
import pandas as pd


base_time = pd.Timestamp("2023-04-20 02:00")

lead_hours = [24, 48, 72, 120, 168]

c = cdsapi.Client()


def folder_of(date_str):
    return f"ERA5{date_str}"


def fix_zip_file(path):
    if not os.path.exists(path):
        return

    if not zipfile.is_zipfile(path):
        return

    print(f"Unzipping {path}")

    tmp = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)

        nc_files = []
        for root, _, files in os.walk(tmp):
            for name in files:
                if name.endswith(".nc"):
                    nc_files.append(os.path.join(root, name))

        if not nc_files:
            raise RuntimeError(f"No nc file found inside {path}")

        os.remove(path)
        shutil.move(nc_files[0], path)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def get_input_times():
    times = []

    for lead in lead_hours:
        t1 = base_time - pd.Timedelta(hours=lead)
        t0 = t1 - pd.Timedelta(hours=6)

        times.append(t0)
        times.append(t1)

    return sorted(set(times))


def tp_dates_for_time(ts):
    ts = pd.to_datetime(ts)

    tp_times = pd.date_range(
        ts - pd.Timedelta(hours=5),
        ts,
        freq="h"
    )

    dates = sorted(set(t.strftime("%Y%m%d") for t in tp_times))

    return dates, tp_times


def download_pressure(ts):
    date = ts.strftime("%Y%m%d")
    hour = ts.strftime("%H")

    folder = folder_of(date)
    os.makedirs(folder, exist_ok=True)

    out = os.path.join(folder, f"P{date}{hour}.nc")

    if os.path.exists(out):
        fix_zip_file(out)
        print(f"Exists: {out}")
        return

    print(f"Downloading pressure: {ts}")

    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "relative_humidity",
            ],
            "pressure_level": [
                "50", "100", "150", "200", "250", "300",
                "400", "500", "600", "700", "850", "925", "1000",
            ],
            "year": date[:4],
            "month": date[4:6],
            "day": date[6:8],
            "time": [f"{hour}:00"],
        },
        out,
    )

    fix_zip_file(out)


def download_surface(ts):
    date = ts.strftime("%Y%m%d")
    hour = ts.strftime("%H")

    folder = folder_of(date)
    os.makedirs(folder, exist_ok=True)

    out = os.path.join(folder, f"S{date}{hour}.nc")

    if os.path.exists(out):
        fix_zip_file(out)
        print(f"Exists: {out}")
        return

    print(f"Downloading surface: {ts}")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "total_precipitation",
            ],
            "year": date[:4],
            "month": date[4:6],
            "day": date[6:8],
            "time": [f"{hour}:00"],
        },
        out,
    )

    fix_zip_file(out)


def download_tp_day(date):
    folder = folder_of(date)
    os.makedirs(folder, exist_ok=True)

    out = os.path.join(folder, f"R{date}.nc")

    if os.path.exists(out):
        fix_zip_file(out)
        print(f"Exists: {out}")
        return

    print(f"Downloading TP day: {date}")

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["total_precipitation"],
            "year": date[:4],
            "month": date[4:6],
            "day": date[6:8],
            "time": [f"{h:02d}:00" for h in range(24)],
        },
        out,
    )

    fix_zip_file(out)


def download_one_time(ts):
    ts = pd.to_datetime(ts)

    print(f"\n--- {ts} ---")

    download_pressure(ts)
    download_surface(ts)

    tp_dates, tp_times = tp_dates_for_time(ts)

    print(f"TP window: {tp_times[0]} to {tp_times[-1]}")

    for date in tp_dates:
        download_tp_day(date)


times = get_input_times()

print("Input times:")
for t in times:
    print(" ", t)

all_tp_dates = set()

for t in times:
    dates, _ = tp_dates_for_time(t)
    all_tp_dates.update(dates)

print("\nTP days needed:")
for d in sorted(all_tp_dates):
    print(" ", d)

for t in times:
    download_one_time(t)

print("\nDone.")