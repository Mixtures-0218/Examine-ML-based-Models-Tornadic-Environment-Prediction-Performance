import os
import numpy as np
import pygrib
import netCDF4 as nc

# CONFIGURATION
# ============================================================================
# Directory containing your downloaded GRIB2 files
grib_dir = "/glade/u/home/lroyce/gfs_forecast_data/target_2023042002/"

# Make sure this directory exists and contains your .grib2 files
if not os.path.exists(grib_dir):
    print(f"Error: Directory not found: {grib_dir}")
    print("Please update grib_dir to point to your GRIB2 files")
    exit(1)


# CONVERT GRIB TO NETCDF FUNCTION
# ============================================================================
def convert_grib_to_netcdf(grib_file, output_nc):
    """Convert GRIB2 to NetCDF - same as in your main script."""
    try:
        print(f"\nConverting: {os.path.basename(grib_file)}")
        
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
        grbs = pygrib.open(grib_file)
        grb = grbs[1]
        lats, lons = grb.latlons()
        nlat, nlon = lats.shape
        
        ncfile = nc.Dataset(output_nc, 'w', format='NETCDF4')
        
        ncfile.createDimension('latitude', nlat)
        ncfile.createDimension('longitude', nlon)
        ncfile.createDimension('level', len(pressure_levels))
        ncfile.createDimension('time', 1)
        
        lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
        lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
        lev_var = ncfile.createVariable('level', 'i4', ('level',))
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        
        lat_var[:] = lats[:, 0]
        lon_var[:] = lons[0, :]
        lev_var[:] = pressure_levels
        time_var[:] = 0
        
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        lev_var.units = 'hPa'
        time_var.units = 'hours since 1900-01-01'
        
        u10_var = ncfile.createVariable('u10', 'f4', ('time', 'latitude', 'longitude'))
        v10_var = ncfile.createVariable('v10', 'f4', ('time', 'latitude', 'longitude'))
        t2m_var = ncfile.createVariable('t2m', 'f4', ('time', 'latitude', 'longitude'))
        msl_var = ncfile.createVariable('msl', 'f4', ('time', 'latitude', 'longitude'))
        
        z_var = ncfile.createVariable('z', 'f4', ('time', 'level', 'latitude', 'longitude'))
        q_var = ncfile.createVariable('q', 'f4', ('time', 'level', 'latitude', 'longitude'))
        u_var = ncfile.createVariable('u', 'f4', ('time', 'level', 'latitude', 'longitude'))
        v_var = ncfile.createVariable('v', 'f4', ('time', 'level', 'latitude', 'longitude'))
        t_var = ncfile.createVariable('t', 'f4', ('time', 'level', 'latitude', 'longitude'))
        
        print("  Extracting surface variables...")
        u10_var[0] = grbs.select(name='10 metre U wind component')[0].values
        v10_var[0] = grbs.select(name='10 metre V wind component')[0].values
        t2m_var[0] = grbs.select(name='2 metre temperature')[0].values
        msl_var[0] = grbs.select(name='Pressure reduced to MSL')[0].values
        
        print("  Extracting upper air variables...")
        for i, level in enumerate(pressure_levels):
            gh = grbs.select(name='Geopotential height', level=level)[0].values
            z_var[0, i] = gh * 9.80665
            
            rh = grbs.select(name='Relative humidity', level=level)[0].values
            temp = grbs.select(name='Temperature', level=level)[0].values
            pressure_pa = level * 100.0
            
            rh_fraction = rh / 100.0
            temp_c = temp - 273.15
            es = 611.2 * np.exp(17.67 * temp_c / (temp - 29.65))
            e = rh_fraction * es
            q = 0.622 * e / (pressure_pa - 0.378 * e)
            q = np.maximum(np.minimum(q, 0.06), 0.0)
            q_var[0, i] = q.astype(np.float32)
            
            u_var[0, i] = grbs.select(name='U component of wind', level=level)[0].values
            v_var[0, i] = grbs.select(name='V component of wind', level=level)[0].values
            t_var[0, i] = temp
        
        grbs.close()
        ncfile.close()
        
        file_size_mb = os.path.getsize(output_nc) / (1024 * 1024)
        print(f"  ✓ Converted successfully ({file_size_mb:.1f} MB)")
        print(f"  Output: {os.path.basename(output_nc)}")
        return True
        
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# MAIN CONVERSION LOOP
# ============================================================================
def main():
    print("=" * 70)
    print("GRIB2 to NetCDF Batch Converter")
    print("=" * 70)
    print(f"\nInput directory: {grib_dir}")
    
    # Find all GRIB2 files
    grib_files = []
    for file in os.listdir(grib_dir):
        if file.endswith('.grib2'):
            grib_files.append(file)
    
    if not grib_files:
        print("\n✗ No GRIB2 files found in directory!")
        print("Make sure your .grib2 files are in the correct location")
        return
    
    print(f"\nFound {len(grib_files)} GRIB2 files to convert:")
    for f in sorted(grib_files):
        print(f"  • {f}")
    
    print(f"\n{'='*70}")
    print("STARTING CONVERSION")
    print(f"{'='*70}")

    successful = 0
    failed = 0
    
    for grib_file in sorted(grib_files):
        grib_path = os.path.join(grib_dir, grib_file)
        
        # Create output filename (replace .grib2 with .nc)
        nc_file = grib_file.replace('.grib2', '.nc')
        nc_path = os.path.join(grib_dir, nc_file)
        
        # Skip if NetCDF already exists
        if os.path.exists(nc_path):
            print(f"\n⚠ Skipping {grib_file} - NetCDF already exists")
            continue
        
        # Convert
        if convert_grib_to_netcdf(grib_path, nc_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total GRIB2 files: {len(grib_files)}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"\nNetCDF files saved in: {grib_dir}")
    
    if successful > 0:
        print(f"\n✓ All converted files have the same format as your original script!")
        print(f"\nYou can now use these .nc files with your analysis tools:")
        print(f"  import xarray as xr")
        print(f"  ds = xr.open_dataset('{grib_dir}/gfs_init_2019052018_f004.nc')")
    
    print("=" * 70)


if __name__ == "__main__":
    main()