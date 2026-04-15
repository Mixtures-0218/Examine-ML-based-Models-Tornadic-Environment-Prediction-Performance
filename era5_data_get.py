import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime, timedelta


# save input_data folder
input_data_dir = os.path.join(
    os.path.join(os.getcwd(), "input_data"),
)
os.makedirs(input_data_dir, exist_ok=True)


def init_time_get():
    print("Please enter the initial time of the data you want to download:")
    valid_time = (datetime.now() - timedelta(days=4)).replace(hour=0, minute=0, second=0)
    print("The available date is before", valid_time.strftime("%Y-%m-%d"), "00:00...")

    while True:
        init_time = datetime(
            year=int(input("year: ")),
            month=int(input("month: ")),
            day=int(input("day: ")),
            hour=int(input("hour (recommend 0, 6, 12, 18): ")),
            minute=0
        )
        if init_time >= valid_time:
            print("The available date is before", valid_time.strftime("%Y-%m-%d"), "00:00...", "Please enter again!")
        else:
            print(f"Downloading ERA5 data for {init_time.strftime('%Y-%m-%d-%H')}" + ":00...")
            return init_time, init_time.strftime('%Y-%m-%d'), init_time.strftime('%H:%M')


def download_era5_data(input_path, init_date, init_hour):
    # required download variables
    surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind',
                         '2m_temperature']
    upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

    # required area
    area = [90, -180, -90, 180]

    # download surface variables
    c = cdsapi.Client()
    c.retrieve('reanalysis-era5-single-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': surface_variables,
                   'date': init_date,
                   'time': init_hour,
                   'area': area
               },os.path.join(input_path, 'input_surface.nc'))
    print("Surface data downloaded successfully")
    print("Start to download upper air data...")

    # download upper air variables
    c.retrieve('reanalysis-era5-pressure-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': upper_variables,
                   'pressure_level': pressure_levels,
                   'date': init_date,
                   'time': init_hour,
                   'area': area
               }, os.path.join(input_path, 'input_upper.nc'))
    print("Upper air data downloaded successfully")


def convert_to_numpy(input_path):
    print("Converting to numpy array...")
    surface_data = np.zeros((4, 721, 1440), dtype = np.float32)
    with nc.Dataset(os.path.join(input_path, 'input_surface.nc')) as nc_file:
        surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
        surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
        surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
        surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
    np.save(os.path.join(input_path, 'input_surface.npy'), surface_data)

    upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
    with nc.Dataset(os.path.join(input_path, 'input_upper.nc')) as nc_file:
        upper_data[0] = nc_file.variables['z'][:].astype(np.float32)
        upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
        upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
        upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
        upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
    np.save(os.path.join(input_path, 'input_upper.npy'), upper_data)
    print("convert to numpy done!")


init_time, init_date, init_hour = init_time_get()
download_era5_data(input_data_dir, init_date, init_hour)
convert_to_numpy(input_data_dir)
