# GFS Data Download
## Introduction
gfs_data_get.py is a Python script that automatically downloads GFS (Global Forecast System) archive data for a user-specified target time. It downloads forecasts from multiple initialization times (exact, 24h, 48h, 72h, 120h, and 168h lead times), all verified against the same target moment, and converts each GRIB2 file into NetCDF format for use in downstream weather AI models such as FengWu

## Background
The purpose of this script is to collect a set of GFS forecasts that all predict the same target time, but were initialized at different points in the past. This allows comparison of forecast skill across different lead times — a key component of severe weather prediction research.

## Requirements
Python Version
-	Python 3.8 or higher is recommended.
Python Packages
Install all dependencies with:
-	pip install numpy pygrib netCDF4 requests

## Purposes:
Numpy - Numerical array operations and specific humidity calculations
Pygrib - Reading GRIB2 meteorological files
netCDF4 - Writing output NetCDF files
Requests - Downloading files from AWS and RDA THREDDS

## pygrib System Dependencies
pygrib requires the ECMWF ecCodes library. Install it before pip install pygrib:
Linux (apt):
bashsudo apt-get install libeccodes-dev
macOS (Homebrew):
bashbrew install eccodes

## Authentication Setup
GFS data availability depends on the year of your target time:

## Year Range – Source – Authentication required?
2021-present – AWS Public S3 – none needed
2015 - 2020 – NCAR RDA – RDA token required


## Getting Your RDA Token (Required for 2015–2020 Data)

1.	Go to https://rda.ucar.edu/ and create a free account.
2.	Log in, then navigate to My Profile (top right corner).
3.	Click "Authentication Token" and copy your token string.
4.	In the same directory as gfs_data_get.py, create a file named rdams_token.txt:

## Running the Script!

python gfs_data_get.py

The script runs interactively and will prompt you to enter your target date and time:

Example:

Please enter the TARGET time (the moment you want to predict):
Year: 2024
Month: 5
Day: 7
Hour (0-23): 2

After entering the time, it will display a summary of all the GFS runs it plans to download and ask for confirmation:
Will download forecasts from these GFS runs:
  1. 2024-05-07 00:00 run → +3h forecast → Target
  2. 2024-05-06 00:00 run → +27h forecast → Target
  3. 2024-05-05 00:00 run → +51h forecast → Target
  ...

Proceed? (y/n):
Type y and press Enter to begin downloading.

## Understanding the Output: All output files are saved to
gfs_forecast_data/target_YYYYMMDDHH/
For example, for a target of May 7, 2024 02:00 UTC:

```plain
gfs_forecast_data/target_2024050702/
├── gfs_init_2024050700_f003.nc     ← Exact (initialized May 7 00Z, +3h)
├── gfs_init_2024050600_f027.nc     ← 24h lead
├── gfs_init_2024050500_f051.nc     ← 48h lead
├── gfs_init_2024050400_f075.nc     ← 72h lead
├── gfs_init_2024050200_f123.nc     ← 120h lead
├── gfs_init_2024043000_f171.nc     ← 168h lead
```

However, sometimes, the data does not get downloaded, even if the data is accessible(?)!

In this case, follow this manual:

Downloading data and converting grib file format to NC
1.	Conda activate myenv
2.	python gfs_data_get.py
3.	Enter the desire dates and follow instructions
4.	If some are missing, this means that the code was not able to retrieve the data from the three sources of the data. If absolutely necessary, you can go to THIS website
a.	Data access → web downloader → File list link → find the desired date and UTC
b.	Press the desired one, then press jupyter notebook download script
c.	Upload to ur jupyter lab, and run the first two cells to download the data.
i.	At this point you have a grib2 file. Go to terminal 
1.	Conda activate myenv
2.	Python convert_GRIB2_NC.py
5.	And now you have the nc files!
6.	IF IT DOES NOT WORK:
a.	Open terminal with the file that has the data 
b.	Conda activate myenv
c.	Copy paste this but replace the file name to the specific file names.

python3 -c "import pygrib; g = pygrib.open('SPECIFIC FILE NAME'); print(g.messages); g.close()"
●	IF IT RETURNS 0, then it is empty or corrupted file

