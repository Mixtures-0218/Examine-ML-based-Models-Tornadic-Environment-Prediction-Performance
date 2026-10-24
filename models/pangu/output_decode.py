import os
import netCDF4 as nc
import numpy as np
import xarray as xr

output_path = os.path.join(os.getcwd(), "output_data")
print(output_path)


def surface_decode(surface_file, file_name, file_path):
    surface_data = np.load(surface_file)
    mslp = surface_data[0]
    u10 = surface_data[1]
    v10 = surface_data[2]
    t2m = surface_data[3]

    with nc.Dataset(
        os.path.join(file_path,file_name), "w", format="NETCDF4"
    )as nc_file:
        nc_file.createDimension("longitude", 1440)
        nc_file.createDimension("latitude", 721)

        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude"))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude"))
        nc_mslp = nc_file.createVariable("mslp", np.float32, ("latitude", "longitude"))
        nc_u10 = nc_file.createVariable("u10", np.float32, ("latitude", "longitude"))
        nc_v10 = nc_file.createVariable("v10", np.float32, ("latitude", "longitude"))
        nc_t2m = nc_file.createVariable("t2m", np.float32, ("latitude", "longitude"))

        nc_lon.unit = "degrees_east"
        nc_lat.unit = "degrees_north"
        nc_mslp.unit = "Pa"
        nc_u10.unit = "m/s"
        nc_v10.unit = "m/s"
        nc_t2m.unit = "K"

        nc_lon[:] = np.linspace(0.125,360,1440)
        nc_lat[:] = np.linspace(90,-90,721)
        nc_mslp[:] = mslp
        nc_u10[:] = u10
        nc_v10[:] = v10
        nc_t2m[:] = t2m


def upper_decode(upper_file, file_name, file_path):
    upper_data = np.load(upper_file)
    geopotential = upper_data[0]
    specific_humidity = upper_data[1]
    temperature = upper_data[2]
    u_component_wind = upper_data[3]
    v_component_wind = upper_data[4]

    with nc.Dataset(
        os.path.join(file_path, file_name),"w", format="NETCDF4"
    )as nc_file:
        nc_file.createDimension("longitude", 1440)
        nc_file.createDimension("latitude", 721)
        nc_file.createDimension("level", 13)

        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude"))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude"))
        nc_geop = nc_file.createVariable("geopotential", np.float32, ("level", "latitude", "longitude"))
        nc_q = nc_file.createVariable("specific_humidity", np.float32, ("level", "latitude", "longitude"))
        nc_t = nc_file.createVariable("temperature", np.float32, ("level", "latitude", "longitude"))
        nc_u = nc_file.createVariable("u_component_wind", np.float32, ("level", "latitude", "longitude"))
        nc_v = nc_file.createVariable("v_component_wind", np.float32, ("level", "latitude", "longitude"))

        nc_lon.unit = "degree_east"
        nc_lat.unit = "degree_north"
        nc_geop.unit = "m"
        nc_q.unit = "kg/kg"
        nc_t.unit = "K"
        nc_u.unit = "m/s"
        nc_v.unit = "m/s"

        nc_lon[:] = np.linspace(0, 359.75, 1440)
        nc_lat[:] = np.linspace(90, -90, 721)
        nc_geop[:] = geopotential
        nc_q[:] = specific_humidity
        nc_t[:] = temperature
        nc_u[:] = u_component_wind
        nc_v[:] = v_component_wind


for file in os.listdir(output_path):
    if file.endswith("surface.npy"):
        surface_decode(os.path.join(output_path, file), "output_surface.nc", output_path)
    elif file.endswith("upper.npy"):
        upper_decode(os.path.join(output_path, file), "output_upper.nc", output_path)
ds = xr.open_dataset(os.path.join(output_path, 'output_upper.nc'))
print(ds["geopotential"].values)