import numpy as np
import os
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import srtm


class tornado_analyze:
    def __init__(self, lonW, lonE, latS, latN, resolution=0.25):
        self.lonW = lonW
        self.lonE = lonE
        self.latS = latS
        self.latN = latN
        self.resolution = resolution

        # Calculate the number of data points
        # Data resolution 0.25
        self.lat_size = int(np.abs(latN - latS) / resolution + 1)
        self.lon_size = int(np.abs(lonE - lonW) / resolution + 1)
        self.layer_name = ['1000', '925', '850', '700', '600', '500',
                           '400', '300', '250', '200', '150', '100', '50']
        self.layer = [1000., 925., 850., 700., 600., 500.,
                      400., 300., 250., 200., 150., 100., 50.] * units.hPa

    def elevation(self):
        # Grid Creating according to region
        lats = np.linspace(self.latN, self.latS, self.lat_size)
        lons = np.linspace(self.lonW, self.lonE, self.lon_size)

        # Download SRTM data
        elevation_data = srtm.get_data()

        # Filling data into grid
        grid = np.empty((self.lat_size, self.lon_size), dtype=float)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                elev = elevation_data.get_elevation(lat, lon)
                grid[i, j] = elev if elev is not None else np.nan
        return grid

    def extraction(self, ds):
        # Generate lat/lon grid
        lat = np.linspace(90, -90, 721)
        lon = np.linspace(0, 360, 1440)

        # Convert longitudes to 0-360 degrees
        lonW_360 = self.lonW % 360  # -100 -> 260
        lonE_360 = self.lonE % 360  # -94 -> 266

        # Find indices
        lat_idx_N = np.argmin(np.abs(lat - self.latN))
        lat_idx_S = np.argmin(np.abs(lat - self.latS))
        lon_idx_W = np.argmin(np.abs(lon - lonW_360))
        lon_idx_E = np.argmin(np.abs(lon - lonE_360))

        # Ensure indices are ordered correctly
        if lat_idx_S > lat_idx_N:
            lat_idx_S, lat_idx_N = lat_idx_N, lat_idx_S
        if lon_idx_W > lon_idx_E:
            lon_idx_W, lon_idx_E = lon_idx_E, lon_idx_W

        # Extract subset
        subset = ds[lat_idx_S:lat_idx_N + 1, lon_idx_W:lon_idx_E + 1]
        return subset

    def convective(self, path, model):
        elevation_grid = self.elevation()
        q_layer = []
        t_layer = []
        print('Processing file: ' + path)
        ds = xr.open_dataset(path)

        # Extract specific humidity and temperature at different layers for dewpoint calculation
        if model == 'real':
            for i in range(13):
                q_layer.append(ds.q.values[0][i] * 1000)  # kg/kg -> g/kg
                t_layer.append(ds.t.values[0][i] - 273.15)  # K -> C
        elif model == 'fengwu_v1' or model == 'fengwu_v2':
            for name in self.layer_name:
                q_layer.append(ds["q" + name] * 1000)
                t_layer.append((ds["t" + name] - 273.15))
        elif model == 'pangu':
            for i in range(13):
                q_layer.append(ds.specific_humidity[i].values * 1000)
                t_layer.append(ds.temperature[i].values - 273.15)
        elif model == 'fuxi':
            # Fuxi missing specific humidity
            # Use relative humidity to calculate dewpoint temperature
            r_layer = []
            for name in self.layer_name:
                r_layer.append(ds["R" + name].values)
                t_layer.append((ds["T" + name].values - 273.15))
        elif model == 'gfs':
            for i in range(12, -1, -1):
                q_layer.append(ds.q.values[0][i] * 1000)  # kg/kg -> g/kg
                t_layer.append(ds.t.values[0][i] - 273.15)  # K -> C
        dt = []
        t_extracted = []

        for t in t_layer:
            t_extracted.append(self.extraction(t))

        if model != 'fuxi':
            # Extract mean sea level pressure data to minimize the influence of the terrain
            if model == 'fengwu_v1' or model == 'fengwu_v2':
                mslp_extracted = self.extraction(ds.mslp.values)
            elif model == 'pangu':
                surface = xr.open_dataset(path[:-3] + '_surface.nc')
                mslp_extracted = self.extraction(surface.mslp.values)
            elif model == 'real':
                surface = xr.open_dataset(path[:-3] + '_surface.nc')
                mslp_extracted = self.extraction(surface.msl.values[0])
            elif model == 'gfs':
                mslp_extracted = self.extraction(ds.msl.values[0])
            # Calculate dewpoint temperature from specific humidity (not fuxi)
            for i, q in enumerate(q_layer):
                dt.append(
                    mpcalc.dewpoint_from_specific_humidity(
                        self.layer[i],
                        self.extraction(q) * units('g/kg')))
        else:
            mslp_extracted = self.extraction(ds.MSL.values)
            # Calculate dewpoint temperature from relative humidity
            for i, r in enumerate(r_layer):
                dt.append(
                    mpcalc.dewpoint_from_relative_humidity(
                        t_extracted[i] * units.degC,
                        self.extraction(r) * units.percent)
                )
        # Transpose dewpoint and temperature to match the input of metpy cape calculation
        trans_dt = np.transpose(dt, (1, 2, 0))
        trans_t_extracted = np.transpose(t_extracted, (1, 2, 0))

        cape = []
        cin = []
        k_idx = []
        lfc = []
        lcl = []
        for lat in range(self.lat_size):
            cape_lon = []
            cin_lon = []
            k_idx_lon = []
            lfc_lon = []
            lcl_lon = []
            for lon in range(self.lon_size):
                # Determine the lowest pressure level according to the terrain
                idx = self.adjusted_pressure(mslp_extracted[lat][lon],
                                             elevation_grid[lat][lon])
                # Calculate CAPE and CIN
                cape_cin = mpcalc.surface_based_cape_cin(
                    self.layer[idx:],
                    trans_t_extracted[lat][lon][idx:] * units.degC,
                    trans_dt[lat][lon][idx:] * units.degC)
                cape_lon.append(cape_cin[0].magnitude)
                cin_lon.append(cape_cin[1].magnitude)
                # Calculate K index
                k_idx_lon.append(mpcalc.k_index(
                    self.layer[idx:],
                    trans_t_extracted[lat][lon][idx:] * units.degC,
                    trans_dt[lat][lon][idx:] * units.degC).magnitude)
                # Calculate LFC
                lfc_lon.append(mpcalc.lfc(
                    self.layer[idx:],
                    trans_t_extracted[lat][lon][idx:] * units.degC,
                    trans_dt[lat][lon][idx:] * units.degC)[0].magnitude)
                # Calculate LCL
                lcl_lon.append(mpcalc.lcl(
                    self.layer[idx],
                    trans_t_extracted[lat][lon][idx] * units.degC,
                    trans_dt[lat][lon][idx] * units.degC)[0].magnitude)
            cape.append(cape_lon)
            cin.append(cin_lon)
            k_idx.append(k_idx_lon)
            lfc.append(lfc_lon)
            lcl.append(lcl_lon)

        cape_array = np.array(cape)
        cin_array = np.array(cin)
        k_idx_array = np.array(k_idx)
        return cape_array, cin_array, k_idx_array, lfc, lcl

    # Determine the index of the lowest pressure level to account for terrain
    def adjusted_pressure(self, mslp, elevation):
        # Calculate the surface pressure due to terrain in hPa
        surface_p = mslp * np.exp(-elevation / 8400) / 100
        # Compare model's lowest pressure level
        # Match the closest pressure level to the surface pressure from terrain
        if np.abs(surface_p - 1000) < np.abs(surface_p - 925):
            return 0  # 1000 hPa
        else:
            return 1  # 925 hPa
        # Oklahoma area is not high enough to use next level 850 hPa

    def helicity(self, path, model):
        u_layer = []
        v_layer = []

        ds = xr.open_dataset(path)

        # Extract U/V wind data from the selected model
        if model == 'real':
            for i in range(13):
                u_layer.append(ds.u.values[0][i])
                v_layer.append(ds.v.values[0][i])
        elif model == 'fengwu_v1' or model == 'fengwu_v2':
            for name in self.layer_name:
                u_layer.append(ds["u" + name])
                v_layer.append(ds["v" + name])
        elif model == 'pangu':
            for i in range(13):
                u_layer.append(ds.u_component_wind[i].values)
                v_layer.append(ds.v_component_wind[i].values)
        elif model == 'fuxi':
            for name in self.layer_name:
                u_layer.append(ds["U" + name])
                v_layer.append(ds["V" + name])
        elif model == 'gfs':
            for i in range(12, -1, -1):
                u_layer.append(ds.u.values[0][i])
                v_layer.append(ds.v.values[0][i])

        MSL = []
        u_extracted = []
        v_extracted = []

        # Calculate Mean Sea Level (MSL) according to pressure
        for p in self.layer:
            MSL.append(mpcalc.pressure_to_height_std(p).to('m'))

        # Calculate elevations for the region
        elevation_value = self.elevation()

        # Calculate Above Ground Level (AGL)
        # according to the difference between elevation and MSL
        AGL_p = []
        for p in MSL:
            AGL = []
            for lat in elevation_value:
                elevation_lat = []
                for lon in lat:
                    elevation_lat.append(p.magnitude - lon)
                AGL.append(elevation_lat)
            AGL_p.append(AGL)

        # Extract U/V wind data for the region
        for u in u_layer:
            u_extracted.append(self.extraction(u))

        for v in v_layer:
            v_extracted.append(self.extraction(v))

        # Transpose the matrices to match the input shape of metpy bunker calculation
        u_trans = np.transpose(u_extracted, (1, 2, 0))
        v_trans = np.transpose(v_extracted, (1, 2, 0))
        AGL_trans = np.transpose(AGL_p, (1, 2, 0))

        # Calculate bunker
        bunker = []
        for lat in range(self.lat_size):
            bunker_lat = []
            for lon in range(self.lon_size):
                bunker_lat.append(
                    mpcalc.bunkers_storm_motion(
                        self.layer,
                        u_trans[lat][lon] *
                        units('m/s'),
                        v_trans[lat][lon] *
                        units('m/s'),
                        AGL_trans[lat][lon] *
                        units.m))
            bunker.append(bunker_lat)

        # Calculate Helicity
        helicity = []
        for lat in range(self.lat_size):
            helicity_lat = []
            for lon in range(self.lon_size):
                helicity_lat.append(mpcalc.storm_relative_helicity(
                    AGL_trans[lat][lon] * units.m,
                    u_trans[lat][lon] * units('m/s'),
                    v_trans[lat][lon] * units('m/s'),
                    depth=3 * units.km,  # Helicity 0-3 km
                    storm_u=bunker[lat][lon][0][0],
                    storm_v=bunker[lat][lon][0][1]))
            helicity.append(helicity_lat)

        # Isolate positive/negative helicity
        positive_helicity = []
        negative_helicity = []
        for lat in range(self.lat_size):
            positive_helicity_lat = []
            negative_helicity_lat = []
            for lon in range(self.lon_size):
                positive_helicity_lat.append(helicity[lat][lon][0].magnitude)
                negative_helicity_lat.append(helicity[lat][lon][1].magnitude)
            positive_helicity.append(positive_helicity_lat)
            negative_helicity.append(negative_helicity_lat)

        return positive_helicity, negative_helicity

    def save(self, model, path, output_path):
        cape = []
        cin = []
        k_idx = []
        lfc = []
        lcl = []
        positive_helicity = []
        negative_helicity = []
        # Calculate tornado parameters for each time file
        for file in path:
            # Calculate cape/cin/k_idx/lfc/lcl
            (
                cape_value,
                cin_value,
                k_idx_value,
                lfc_value,
                lcl_value,
            ) = self.convective(file, model)
            cape.append(cape_value)
            cin.append(cin_value)
            k_idx.append(k_idx_value)
            lfc.append(lfc_value)
            lcl.append(lcl_value)
            # Calculate helicity
            positive, negative = self.helicity(file, model)
            positive_helicity.append(positive)
            negative_helicity.append(negative)

        # Saving and naming
        if model == 'real':
            # Real file doesn't have time dimension
            np.save(os.path.join(output_path, "cape_real.npy"), cape_value)
            np.save(os.path.join(output_path, "cin_real.npy"), cin_value)
            np.save(os.path.join(output_path, "k_idx_real.npy"), k_idx_value)
            np.save(
                os.path.join(
                    output_path,
                    "positive_helicity_real.npy"),
                positive)
            np.save(
                os.path.join(
                    output_path,
                    "negative_helicity_real.npy"),
                negative)
            np.save(os.path.join(output_path, "lfc_real.npy"), lfc_value)
            np.save(os.path.join(output_path, "lcl_real.npy"), lcl_value)
        else:
            np.save(
                os.path.join(
                    output_path,
                    f"cape_{model}.npy"),
                np.array(cape))
            np.save(
                os.path.join(
                    output_path,
                    f"cin_{model}.npy"),
                np.array(cin))
            np.save(
                os.path.join(
                    output_path,
                    f"k_idx_{model}.npy"),
                np.array(k_idx))
            np.save(
                os.path.join(
                    output_path,
                    f"positive_helicity_{model}.npy"),
                np.array(positive_helicity))
            np.save(
                os.path.join(
                    output_path,
                    f"negative_helicity_{model}.npy"),
                np.array(negative_helicity))
            np.save(
                os.path.join(
                    output_path,
                    f"lfc_{model}.npy"),
                np.array(lfc))
            np.save(
                os.path.join(
                    output_path,
                    f"lcl_{model}.npy"),
                np.array(lcl))
        return f'{model} parameters saved!'


# models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi', 'real']
models = ['gfs']

# Define bounds (Oklahoma region)
lonW, lonE = -100.0, -94.0
latS, latN = 33.0, 38.0

# All EF3

"""times = ['2013-05-20-20UTC', '2013-05-31-23UTC', '2015-05-07-02UTC',
         '2015-05-25-20UTC', '2016-05-09-21UTC', '2023-04-20-02UTC',
         '2024-04-28-04UTC', '2024-05-07-02UTC', '2024-05-26-04UTC',
         '2025-05-19-23UTC']"""
times = ['2025-05-19-23UTC']

for time in times:
    # Select forecast time
    # Change time here for different event!
    parent_path = os.path.join(os.getcwd(), time)
    # Choose different models and folders
    for model in models:
        if model == 'real':
            path = [os.path.join(parent_path, 'real.nc')]
            output_path = parent_path
        elif model == 'gfs':
            path = []
            sub_path = os.path.join(parent_path, 'gfs')
            output_path = sub_path
            for file in os.listdir(sub_path):
                if file.endswith('.nc') and '._' not in file:
                    path.append(os.path.join(sub_path, file))
            path = sorted(path)[::-1]
        else:
            sub_path = os.path.join(parent_path, model)
            output_path = sub_path
            path = [os.path.join(sub_path, f'{model}_24.nc'),
                    os.path.join(sub_path, f'{model}_48.nc'),
                    os.path.join(sub_path, f'{model}_72.nc'),
                    os.path.join(sub_path, f'{model}_120.nc'),
                    os.path.join(sub_path, f'{model}_168.nc')]
        # Run analyzer with different model outputs
        analyzer = tornado_analyze(lonW, lonE, latS, latN, resolution=0.25)
        saving = analyzer.save(model, path, output_path)
        print(saving)
