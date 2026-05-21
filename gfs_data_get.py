import os
import sys
import requests
import numpy as np
import pygrib
from datetime import datetime, timedelta
import shutil


# CONFIGURATION
# ============================================================================
output_data_dir = os.path.join(os.getcwd(), "gfs_forecast_data")
os.makedirs(output_data_dir, exist_ok=True)

# RDA Token file
TOKEN_FILE = 'rdams_token.txt'

# Lead times in hours - how far BEFORE the target time each forecast was initiated
LEAD_TIMES = [24, 48, 72, 120, 168]

# GLADE path (will check if accessible)
GLADE_GFS_PATH = "/glade/campaign/collections/rda/data/ds084.1"


# AUTHENTICATION
# ============================================================================
def get_authentication():
    """Read RDA authentication token from file."""
    try:
        with open(TOKEN_FILE, 'r') as f:
            token = f.read().strip()
        
        if not token:
            print(f"Error: {TOKEN_FILE} is empty!")
            print("Please add your RDA authentication token to the file.")
            return None
        
        print(f"✓ Token loaded from {TOKEN_FILE}")
        return token
        
    except FileNotFoundError:
        print(f"Note: {TOKEN_FILE} not found")
        print("Will try downloads without RDA authentication (AWS only)")
        print("\nTo enable RDA downloads for 2015-2020 data:")
        print("1. Go to https://rda.ucar.edu/")
        print("2. Sign in → 'My Profile' → get Authentication Token")
        print("3. Save token in 'rdams_token.txt'")
        return None


# HELPER FUNCTION: FIND CLOSEST GFS RUN BEFORE A TIME
# ============================================================================
def find_closest_gfs_run_before(target_time):
    """
    Find the closest GFS run at or before the target time.
    
    GFS runs at 00, 06, 12, 18 UTC.
    For example:
    - Target: 2024-04-27 04:00 UTC
    - Closest run at or before: 2024-04-27 00:00 UTC
    
    Args:
        target_time: Target datetime
        
    Returns:
        datetime: GFS run time at or before target
    """
    gfs_runs = [0, 6, 12, 18]
    target_hour = target_time.hour
    
    # Find the closest run hour at or before target hour
    run_candidates = [h for h in gfs_runs if h <= target_hour]
    
    if run_candidates:
        # Use the latest run on the same day
        run_hour = max(run_candidates)
        run_time = target_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    else:
        # Target hour is before 00:00 UTC, use previous day's 18:00 UTC run
        run_time = (target_time - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    
    return run_time


# USER INPUT FUNCTION
# ============================================================================
def get_target_time():
    """Get target time from user input."""
    print("=" * 70)
    print("GFS FORECAST VERIFICATION DOWNLOADER")
    print("=" * 70)
    print("\nThis tool downloads multiple GFS forecasts that all predict the")
    print("SAME target time, but were initiated at different times.")
    print("\nFor example, target = 2024-04-27 04:00 UTC will download:")
    print("  • Analysis from 2024-04-27 00:00 run (+4h forecast)")
    print("  • Forecast from 2024-04-26 00:00 run (+28h forecast)")
    print("  • Forecast from 2024-04-25 00:00 run (+52h forecast)")
    print("  • Forecast from 2024-04-24 00:00 run (+76h forecast)")
    print("  • Forecast from 2024-04-22 00:00 run (+124h forecast)")
    print("  • Forecast from 2024-04-20 00:00 run (+172h forecast)")
    print("\nAll files predict the same moment in time!")
    print("=" * 70)
    
    print("\nPlease enter the TARGET time (the moment you want to predict):")
    print("(GFS archive available from 2015-01-15 onwards)")
    print("AWS public data: 2021+ (no authentication)")
    print("RDA data: 2015-2020 (requires authentication token)\n")
    
    while True:
        try:
            target_time = datetime(
                year=int(input("Year: ")),
                month=int(input("Month: ")),
                day=int(input("Day: ")),
                hour=int(input("Hour (0-23): ")),
                minute=0
            )
            
            if target_time < datetime(2015, 1, 15):
                print("\nError: GFS data only available from 2015-01-15 onwards")
                continue
            
            # Check if too recent (need about 3 days latency)
            too_recent = datetime.now() - timedelta(days=3)
            if target_time > too_recent:
                print(f"\nWarning: Data from the last 3 days may not be available yet")
                print(f"Latest safe date: {too_recent.strftime('%Y-%m-%d')}")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    continue
            
            # Find the closest GFS run at or before the target
            closest_run = find_closest_gfs_run_before(target_time)
            hours_from_closest = int((target_time - closest_run).total_seconds() / 3600)
            
            print(f"\n{'='*70}")
            print(f"TARGET TIME: {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"{'='*70}")
            print(f"\nClosest GFS run at or before target: {closest_run.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"This run is +{hours_from_closest}h before reaching target\n")
            
            print("Will download forecasts from these GFS runs:")
            print(f"  1. {closest_run.strftime('%Y-%m-%d %H:%M')} run → +{hours_from_closest}h forecast → Target")
            
            for i, lead in enumerate(LEAD_TIMES, 2):
                # Go back 'lead' hours from target, then find the GFS run
                init_time = target_time - timedelta(hours=lead)
                
                # Find closest GFS run at or before this initialization time
                gfs_run = find_closest_gfs_run_before(init_time)
                
                # Calculate forecast hour from this run to target
                forecast_hour = int((target_time - gfs_run).total_seconds() / 3600)
                
                print(f"  {i}. {gfs_run.strftime('%Y-%m-%d %H:%M')} run → +{forecast_hour}h forecast → Target")
            
            print(f"{'='*70}")
            
            if input("\nProceed? (y/n): ").lower() == 'y':
                return target_time
                
        except ValueError:
            print("Invalid input. Try again.\n")


# GET GFS FILE FOR A SPECIFIC RUN AND FORECAST HOUR
# ============================================================================
def get_gfs_file(run_time, forecast_hour, token, output_dir, description):
    """
    Download a specific GFS forecast file.
    
    LOGIC:
    - run_time: When GFS was initiated (e.g., 2024-04-26 00:00 UTC)
    - forecast_hour: How many hours ahead to forecast (e.g., +28h)
    - This gives us the valid time: run_time + forecast_hour
    
    Example:
    - run_time = 2024-04-26 00:00 UTC
    - forecast_hour = 28
    - valid_time = 2024-04-27 04:00 UTC (our target!)
    
    Args:
        run_time: GFS run initialization time
        forecast_hour: Forecast lead time in hours
        token: RDA authentication token (or None)
        output_dir: Output directory
        description: Description for logging
    
    Returns:
        Path to downloaded file or None
    """
    # Calculate valid time
    valid_time = run_time + timedelta(hours=forecast_hour)
    
    # File info
    year = run_time.strftime('%Y')
    date = run_time.strftime('%Y%m%d')
    hour = run_time.strftime('%H')
    forecast_str = f"{forecast_hour:03d}"
    
    # RDA filename format
    rda_filename = f"gfs.0p25.{date}{hour}.f{forecast_str}.grib2"
    
    # Output filename - shows which run this is from
    output_filename = f"gfs_init_{date}{hour}_f{forecast_str}.grib2"
    output_path = os.path.join(output_dir, output_filename)
    
    # Print info
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"GFS Run (Init): {run_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Forecast hour: +{forecast_hour}h")
    print(f"Valid time: {valid_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"File: {rda_filename}")
    print(f"{'='*70}")
    
    # METHOD 1: Check GLADE (if accessible)
    glade_file = os.path.join(GLADE_GFS_PATH, year, date, rda_filename)
    if os.path.exists(glade_file):
        print(f"✓ Found on GLADE: {glade_file}")
        print(f"  Copying to: {output_path}")
        try:
            shutil.copy2(glade_file, output_path)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ Copied successfully ({file_size_mb:.1f} MB)")
            return output_path
        except Exception as e:
            print(f"  ✗ Copy failed: {e}")
    else:
        print(f"  GLADE file not found (or no GLADE access)")
    
    # Determine which method to try based on date
    run_date = datetime(int(year), int(run_time.strftime('%m')), int(run_time.strftime('%d')))
    
    # METHOD 2: Try AWS (for 2021+ data) - NO AUTHENTICATION NEEDED
    if run_date >= datetime(2021, 1, 1):
        print(f"\n  Attempting download from AWS (2021+ data)...")
        
        aws_urls = [
            # Newer structure (2021+)
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{forecast_str}",
            # Alternative structure
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{hour}/gfs.t{hour}z.pgrb2.0p25.f{forecast_str}",
        ]
        
        for aws_url in aws_urls:
            print(f"  Trying: {aws_url}")
            
            try:
                response = requests.get(aws_url, stream=True, timeout=180)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
                    
                    print()
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  ✓ Downloaded from AWS ({file_size_mb:.1f} MB)")
                    return output_path
                
                elif response.status_code == 404:
                    print(f"  404 - Not found, trying next URL...")
                    continue
                else:
                    print(f"  Status {response.status_code}, trying next...")
                    continue
                    
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if os.path.exists(output_path):
            os.remove(output_path)
    
    # METHOD 3: Try RDA THREDDS (for all data, but requires authentication)
    if token:
        print(f"\n  Attempting download from RDA THREDDS...")
        
        rda_urls = [
            f"https://data.rda.ucar.edu/ds084.1/{year}/{date}/{rda_filename}",
            f"https://thredds.rda.ucar.edu/thredds/fileServer/files/g/ds084.1/{year}/{date}/{rda_filename}",
            f"https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/{year}/{date}/{rda_filename}",
        ]
        
        headers = {'Authorization': f'Bearer {token}'}
        
        for rda_url in rda_urls:
            print(f"  Trying: {rda_url}")
            
            try:
                response = requests.get(rda_url, headers=headers, stream=True, timeout=180)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
                    
                    print()
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  ✓ Downloaded from RDA ({file_size_mb:.1f} MB)")
                    return output_path
                
                elif response.status_code == 404:
                    print(f"  404 - Not found, trying next URL...")
                    continue
                elif response.status_code == 401:
                    print(f"  401 - Authentication failed")
                    break
                else:
                    print(f"  Status {response.status_code}, trying next...")
                    continue
                    
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        print(f"\n  Skipping RDA (no authentication token)")
        if run_date < datetime(2021, 1, 1):
            print(f"  Note: For {year} data, RDA authentication is required!")
    
    print(f"\n✗ Could not obtain file from any source")
    return None


# CONVERT GRIB TO NETCDF
# ============================================================================
def convert_grib_to_netcdf(grib_file, output_nc):
    """Convert GRIB2 to NetCDF."""
    try:
        import netCDF4 as nc
        
        print(f"\n  Converting to NetCDF: {os.path.basename(output_nc)}")
        
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
        
        u10_var[0] = grbs.select(name='10 metre U wind component')[0].values
        v10_var[0] = grbs.select(name='10 metre V wind component')[0].values
        t2m_var[0] = grbs.select(name='2 metre temperature')[0].values
        msl_var[0] = grbs.select(name='Pressure reduced to MSL')[0].values
        
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
        
        print(f"  ✓ NetCDF created")
        return output_nc
        
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        return None


# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("GFS Forecast Verification Data Downloader")
    print("=" * 70)
    
    # Get RDA token if available
    token = get_authentication()
    
    # Check GLADE access
    has_glade = os.path.exists(GLADE_GFS_PATH)
    if has_glade:
        print(f"✓ GLADE access detected")
    else:
        print(f"  No GLADE access")
    
    print(f"\nDownload priority:")
    if has_glade and token:
        print(f"  1. GLADE (local) → 2. AWS (2021+) → 3. RDA (all years)")
    elif has_glade:
        print(f"  1. GLADE (local) → 2. AWS (2021+)")
    elif token:
        print(f"  1. AWS (2021+) → 2. RDA (all years)")
    else:
        print(f"  1. AWS (2021+ only)")
    print()
    
    target_time = get_target_time()
    
    target_dir = os.path.join(
        output_data_dir,
        f"target_{target_time.strftime('%Y%m%d%H')}"
    )
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# DOWNLOADING FORECASTS FOR TARGET: {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'#'*70}")
    
    successful = 0
    failed = 0
    files_downloaded = []
    
    # Calculate all the runs we need
    runs_to_download = []
    
    # 1. Closest run at or before target (shortest forecast)
    closest_run = find_closest_gfs_run_before(target_time)
    hours_from_closest = int((target_time - closest_run).total_seconds() / 3600)
    runs_to_download.append((closest_run, hours_from_closest, "Closest run"))
    
    # 2. Runs from 24h, 48h, 72h, 120h, 168h before target
    for lead in LEAD_TIMES:
        # Go back 'lead' hours from target
        init_time = target_time - timedelta(hours=lead)
        
        # Find closest GFS run at or before this time
        gfs_run = find_closest_gfs_run_before(init_time)
        
        # Calculate forecast hour from this run to target
        forecast_hour = int((target_time - gfs_run).total_seconds() / 3600)
        
        runs_to_download.append((gfs_run, forecast_hour, f"~{lead}h before target"))
    
    total = len(runs_to_download)
    
    # Download each forecast
    for idx, (run_time, forecast_hour, desc) in enumerate(runs_to_download, 1):
        print(f"\n[{idx}/{total}] {desc}")
        
        grib_path = get_gfs_file(run_time, forecast_hour, token, target_dir, desc)
        
        if grib_path:
            # Create NetCDF with descriptive name
            date_str = run_time.strftime('%Y%m%d%H')
            nc_filename = f"gfs_init_{date_str}_f{forecast_hour:03d}.nc"
            nc_path = os.path.join(target_dir, nc_filename)
            
            if convert_grib_to_netcdf(grib_path, nc_path):
                successful += 1
                files_downloaded.append((
                    f"Init: {run_time.strftime('%Y-%m-%d %H:%M')} +{forecast_hour}h",
                    nc_path
                ))
            else:
                failed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Target Time: {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"\nOutput directory: {target_dir}")
    
    if files_downloaded:
        print(f"\nAll files below predict: {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Downloaded forecasts:")
        for desc, filepath in files_downloaded:
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {desc}: {os.path.basename(filepath)} ({file_size:.1f} MB)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
    