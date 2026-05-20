# Pangu Weather Inference Process

## The Model files should be downloaded from the original GitHub Page
https://github.com/198808xc/Pangu-Weather

## How to get inference results
1. Run file 'inference_gpu.py'. This file will automatically run the 'era5_data_get.py' file to ask the user to input an initialization date
2. Once all the ERA5 data of the initialization date has been downloaded, the 'inference_gpu.py' will ask the user to input a forecast date
3. The script will automatically compute the inference step and start to inference
4. Once the inference has finished, run the file 'output_decode.py' to decode the result '.npy' files into '.nc' files
