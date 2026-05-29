import numpy as np
import os
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from tornado_parameters import tornado_analyze


def barometric_height(pressure_hPa):
    """
    Convert pressure (hPa) to height (m) using the standard barometric formula.

    h = 44330.8 * (1 - (p / 1013.25) ** (1 / 5.255))
    """
    height_m = 44330.8 * (1 - (pressure_hPa / 1013.25) ** (1 / 5.255))
    return height_m


def calc_bulk_shear_6km(u_crop, v_crop, MSL, elev_grid, lat_size, lon_size):
    """
    Calculate 0-6 km bulk wind shear for each grid point.

    Parameters
    ----------
    u_crop : list of 2D arrays (lat, lon), length 13
        U-wind component at each pressure level, already cropped.
    v_crop : list of 2D arrays (lat, lon), length 13
        V-wind component at each pressure level, already cropped.
    MSL : list of quantities, length 13
        Standard atmosphere height (MSL) for each pressure level.
    elev_grid : 2D array (lat, lon)
        Terrain elevation.
    lat_size, lon_size : int
        Spatial grid dimensions.

    Returns
    -------
    shear : 2D array (lat, lon)
        0-6 km bulk wind shear magnitude (m/s).
    """
    n_levels = len(MSL)
    AGL_p = []
    for p_height in MSL:
        AGL_layer = []
        for lat in range(lat_size):
            lon_row = []
            for lon in range(lon_size):
                lon_row.append(p_height.magnitude - elev_grid[lat, lon])
            AGL_layer.append(lon_row)
        AGL_p.append(AGL_layer)

    u_trans = np.transpose(u_crop, (1, 2, 0))
    v_trans = np.transpose(v_crop, (1, 2, 0))
    AGL_trans = np.transpose(AGL_p, (1, 2, 0))

    shear = np.zeros((lat_size, lon_size))
    for lat in range(lat_size):
        for lon in range(lon_size):
            heights = AGL_trans[lat, lon, :]
            u_prof = u_trans[lat, lon, :]
            v_prof = v_trans[lat, lon, :]

            # Find bottom level: lowest level above terrain (AGL > 0)
            bottom_idx = None
            for i in range(n_levels):
                if heights[i] > 0:
                    bottom_idx = i
                    break

            # Find top level: closest to 6000 m AGL
            top_idx = None
            min_diff = 99999.0
            for i in range(n_levels):
                diff = abs(heights[i] - 6000.0)
                if diff < min_diff:
                    min_diff = diff
                    top_idx = i

            if bottom_idx is not None and top_idx is not None and bottom_idx < top_idx:
                u_diff = u_prof[top_idx] - u_prof[bottom_idx]
                v_diff = v_prof[top_idx] - v_prof[bottom_idx]
                shear[lat, lon] = np.sqrt(u_diff ** 2 + v_diff ** 2)
            else:
                shear[lat, lon] = 0.0

    return shear


def build_model_paths(model, parent_path):
    """Build list of NetCDF file paths for a given model."""
    if model == 'real':
        return [os.path.join(parent_path, 'real.nc')]
    elif model == 'gfs':
        sub_path = os.path.join(parent_path, 'gfs')
        paths = []
        for file in sorted(os.listdir(sub_path)):
            if file.endswith('.nc') and '._' not in file:
                paths.append(os.path.join(sub_path, file))
        return paths[::-1]  # match order of .npy files
    else:
        sub_path = os.path.join(parent_path, model)
        leads = ['24', '48', '72', '120', '168']
        return [os.path.join(sub_path, f'{model}_{lead}.nc') for lead in leads]


def read_wind_data(ds, model, layer_name):
    """
    Read 13-level U/V wind data from a NetCDF dataset.

    Returns
    -------
    u_raw, v_raw : list of 2D arrays (721, 1440) or 3D with single time
    """
    u_raw = []
    v_raw = []
    if model == 'real':
        for i in range(13):
            u_raw.append(ds.u.values[0][i])
            v_raw.append(ds.v.values[0][i])
    elif model == 'pangu':
        for i in range(13):
            u_raw.append(ds.u_component_wind[i].values)
            v_raw.append(ds.v_component_wind[i].values)
    elif model == 'fengwu_v1' or model == 'fengwu_v2':
        for name in layer_name:
            u_raw.append(ds["u" + name])
            v_raw.append(ds["v" + name])
    elif model == 'fuxi':
        for name in layer_name:
            u_raw.append(ds["U" + name])
            v_raw.append(ds["V" + name])
    elif model == 'gfs':
        # GFS stores levels top-to-bottom [50..1000], reverse to [1000..50]
        for i in range(12, -1, -1):
            u_raw.append(ds.u.values[0][i])
            v_raw.append(ds.v.values[0][i])
    return u_raw, v_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # All EF3 tornado events
    times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
             '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
             '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
             '2025-05-19-23UTC']

    models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi', 'real']

    # Define bounds (Oklahoma region)
    lonW, lonE = -100.0, -94.0
    latS, latN = 33.0, 38.0

    # Create analyzer instance for cropping and elevation
    analyzer = tornado_analyze(lonW, lonE, latS, latN, resolution=0.25)
    elev_grid = analyzer.elevation()

    # Standard atmosphere height (MSL) for each pressure level
    MSL = []
    for p in analyzer.layer:
        MSL.append(mpcalc.pressure_to_height_std(p).to('m'))

    print('=== STP (Significant Tornado Parameter) Calculation ===')
    for time in times:
        parent_path = os.path.join(os.getcwd(), time)
        print(f'\nProcessing event: {time}')
        for model in models:
            print(f'  Model: {model}')
            paths = build_model_paths(model, parent_path)

            # Determine output directory
            if model == 'real':
                output_path = parent_path
                model_dir = parent_path
            else:
                model_dir = os.path.join(parent_path, model)
                output_path = model_dir

            # Load pre-computed parameters from .npy files
            cape = np.load(os.path.join(model_dir, f'cape_{model}.npy'))
            srh = np.load(os.path.join(model_dir, f'positive_helicity_{model}.npy'))
            lcl_hPa = np.load(os.path.join(model_dir, f'lcl_{model}.npy'))

            # Convert LCL from hPa to m
            lcl_height = barometric_height(lcl_hPa)

            # Compute 0-6 km bulk shear for each lead time file
            shear_list = []
            for file_idx, path in enumerate(paths):
                print(f'    Processing file: {os.path.basename(path)}')
                ds = xr.open_dataset(path)
                u_raw, v_raw = read_wind_data(ds, model, analyzer.layer_name)
                ds.close()

                # Crop global wind data to the study region
                u_crop = []
                v_crop = []
                for u_level in u_raw:
                    u_crop.append(analyzer.extraction(u_level))
                for v_level in v_raw:
                    v_crop.append(analyzer.extraction(v_level))

                # Calculate bulk shear
                shear_field = calc_bulk_shear_6km(
                    u_crop, v_crop, MSL, elev_grid,
                    analyzer.lat_size, analyzer.lon_size
                )
                shear_list.append(shear_field)

            # Stack shear fields
            if model == 'real':
                shear_stack = np.array(shear_list[0])  # (21, 25)
            else:
                shear_stack = np.array(shear_list)  # (5, 21, 25)

            # Compute STP (Fixed-Layer SB variant)
            # Component 1: SBCAPE / 1500
            cape_term = cape / 1500.0
            cape_term[cape <= 0] = 0.0

            # Component 2: (2000 - LCL) / 1000, with clipping
            lcl_term = (2000.0 - lcl_height) / 1000.0
            lcl_term[lcl_height < 1000.0] = 1.0
            lcl_term[lcl_height > 2000.0] = 0.0

            # Component 3: SRH_0-3km / 100
            srh_term = srh / 100.0
            srh_term[srh <= 0] = 0.0

            # Component 4: Shear_6km / 20, with clipping
            shear_term = shear_stack / 20.0
            shear_term[shear_stack < 12.5] = 0.0
            shear_term[shear_stack > 30.0] = 1.5

            # Final STP
            stp = cape_term * lcl_term * srh_term * shear_term
            stp = np.maximum(stp, 0.0)

            # Handle NaN (e.g., from invalid LCL or LFC not used)
            stp = np.nan_to_num(stp, nan=0.0)

            # Save
            out_file = os.path.join(output_path, f'stp_{model}.npy')
            np.save(out_file, stp)
            print(f'    Saved: {out_file}')
            if model == 'real':
                print(f'      shape={stp.shape}, min={np.min(stp):.4f}, max={np.max(stp):.4f}, mean={np.mean(stp):.4f}')
            else:
                for t_idx in range(stp.shape[0]):
                    print(f'      lead[{t_idx}]: min={np.min(stp[t_idx]):.4f}, max={np.max(stp[t_idx]):.4f}, mean={np.mean(stp[t_idx]):.4f}')

    print('\n=== STP calculation complete! ===')
