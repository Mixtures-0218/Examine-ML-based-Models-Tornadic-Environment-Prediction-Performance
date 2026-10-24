#import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime, timedelta


# save input_data folder
input_data_dir = os.path.join(
    os.path.join(os.getcwd(), "input_data"),
)
os.makedirs(input_data_dir, exist_ok=True)


def init_time_get(target_time):
    print("Please enter the initial time of the data you want to download:")
    valid_time = (datetime.now() - timedelta(days=5)).replace(hour=0, minute=0, second=0)
    # ERA5 data has 5 days latency
    print("The available date is before", valid_time.strftime("%Y-%m-%d"), "00:00...")

    # Entering a date for data
    while True:
        '''
        init_time = datetime(
            year=int(input("year: ")),
            month=int(input("month: ")),
            day=int(input("day: ")),
            hour=int(input("hour (recommend 0, 6, 12, 18): ")),
            minute=0
        )
        '''
        init_time = target_time
        if init_time >= valid_time:
            print("The available date is before", valid_time.strftime("%Y-%m-%d"), "00:00...", "Please enter again!")
        else:
            print(f"Downloading ERA5 data for {init_time.strftime('%Y-%m-%d-%H')}" + ":00...")
            init_time_6h = init_time - timedelta(hours=6) # FengWu requires for additional data after 6h
            return init_time, init_time.strftime('%Y-%m-%d'), init_time.strftime('%H:%M'), init_time_6h.strftime('%Y-%m-%d'), init_time_6h.strftime('%H:%M')


def download_era5_data(input_path, init_date, init_hour, init_date_6h, init_hour_6h, lead_time):
    # required download variables
    surface_variables = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                         '2m_temperature','mean_sea_level_pressure',]
    upper_variables = ['geopotential', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'temperature']
    pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

    # required area
    area = [90, 0, -90, 360]


    # download variables for input1
    c = cdsapi.Client()
    print("Start to download surface data +0...")
    c.retrieve('reanalysis-era5-single-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': surface_variables,
                   'date': init_date,
                   'time': init_hour,
                   'area': area
               },os.path.join(input_path, f'input_surface2_{lead_time}.nc'))
    print("Surface data +0 downloaded successfully")
    print("Start to download upper air data +0...")

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
               }, os.path.join(input_path, f'input_upper2_{lead_time}.nc'))
    print("Upper air data +0 downloaded successfully")
    
    # download variables for input1
    print("Start to download surface data -6h...")
    c.retrieve('reanalysis-era5-single-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': surface_variables,
                   'date': init_date_6h,
                   'time': init_hour_6h,
                   'area': area
               },os.path.join(input_path, f'input_surface1_{lead_time}.nc'))
    print("Surface data -6 downloaded successfully")
    print("Start to download upper air data -6h...")

    # download upper air variables
    c.retrieve('reanalysis-era5-pressure-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': upper_variables,
                   'pressure_level': pressure_levels,
                   'date': init_date_6h,
                   'time': init_hour_6h,
                   'area': area
               }, os.path.join(input_path, f'input_upper1_{lead_time}.nc'))
    print("Upper air data -6h downloaded successfully")


def convert_to_numpy(input_path, lead_time):
    print("Converting to numpy array for input1...")
    input_data_1 = np.zeros((69, 721, 1440), dtype = np.float32) 
    # FengWu's dimension is 69, 721, 1440 include surface and upper data
    with nc.Dataset(os.path.join(input_path, f'input_surface1_{lead_time}.nc')) as nc_surface_file:
        input_data_1[0] = nc_surface_file.variables['u10'][:].astype(np.float32)
        input_data_1[1] = nc_surface_file.variables['v10'][:].astype(np.float32)
        input_data_1[2] = nc_surface_file.variables['t2m'][:].astype(np.float32)
        input_data_1[3] = nc_surface_file.variables['msl'][:].astype(np.float32)
    with nc.Dataset(os.path.join(input_path, f'input_upper1_{lead_time}.nc')) as nc_upper_file:
        variables = ['z', 'q', 'u', 'v', 't'] # Upper data
        flag = 4
        for var in variables:
            data = np.flip(nc_upper_file.variables[var][0,: 13,: ,: ], axis=0)
            # FengWu requires for inverse sorting of pressure level (from top to bottom)
            for i in data:
                input_data_1[flag] = i
                flag += 1
    np.save(os.path.join(input_path, f'input1_{lead_time}.npy'), input_data_1)

    #Data for 6h after
    print("Converting to numpy array for input2...")
    input_data_2 = np.zeros((69, 721, 1440), dtype = np.float32)
    with nc.Dataset(os.path.join(input_path, f'input_surface2_{lead_time}.nc')) as nc_surface_file:
        input_data_2[0] = nc_surface_file.variables['u10'][:].astype(np.float32)
        input_data_2[1] = nc_surface_file.variables['v10'][:].astype(np.float32)
        input_data_2[2] = nc_surface_file.variables['t2m'][:].astype(np.float32)
        input_data_2[3] = nc_surface_file.variables['msl'][:].astype(np.float32)
    with nc.Dataset(os.path.join(input_path, f'input_upper2_{lead_time}.nc')) as nc_upper_file:
        variables = ['z', 'q', 'u', 'v', 't']
        flag = 4
        for var in variables:
            data = np.flip(nc_upper_file.variables[var][0,: 13,: ,: ], axis=0)
            for i in data:
                input_data_2[flag] = i
                flag += 1
    np.save(os.path.join(input_path, f'input2_{lead_time}.npy'), input_data_2)
    print("convert to numpy done!")

lead_times = [24, 48, 72, 120, 168]
init_time_list = []
for lead_time in lead_times:
    target_time = datetime(
                year=2015,
                month=5,
                day=7,
                hour=2,
                minute=0
            )
    target_time = target_time - timedelta(hours=lead_time)
    init_time_list.append(target_time)
    init_time, init_date, init_hour, init_date_6h, init_hour_6h = init_time_get(target_time)
    # download_era5_data(input_data_dir, init_date, init_hour, init_date_6h, init_hour_6h, lead_time)
    # convert_to_numpy(input_data_dir, lead_time)
