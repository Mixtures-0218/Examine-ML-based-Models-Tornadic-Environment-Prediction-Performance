import os
import numpy as np
import onnx
import onnxruntime as ort
import time
from datetime import datetime
from era5_data_get import init_time

print("Please input the forecast time")
forecast_time = datetime(
    year=int(input("year: ")),
    month=int(input("month: ")),
    day=int(input("day: ")),
    hour=int(input("hour")),
    minute=0)

hour_diff = (forecast_time - init_time).total_seconds() / 3600
print(init_time, forecast_time, hour_diff)

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model_24 = onnx.load('pangu_weather_24.onnx')
model_6 = onnx.load('pangu_weather_6.onnx')
model_3 = onnx.load('pangu_weather_3.onnx')
model_1 = onnx.load('pangu_weather_1.onnx')


# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 32
options.inter_op_num_threads = 8

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_6 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_3 = ort.InferenceSession('pangu_weather_3.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_1 = ort.InferenceSession('pangu_weather_1.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

# Run the inference session
start_time = time.time()
print("Start to inference data...")

while hour_diff > 0:
    if hour_diff >= 24:
        output_upper, output_surface = ort_session_24.run(None, {'input': input_upper, 'input_surface': input_surface})
        hour_diff -= 24
    elif hour_diff >= 6:
        output_upper, output_surface = ort_session_6.run(None, {'input': input_upper, 'input_surface': input_surface})
        hour_diff -= 6
    elif hour_diff >= 3:
        output_upper, output_surface = ort_session_3.run(None, {'input': input_upper, 'input_surface': input_surface})
        hour_diff -= 3
    elif hour_diff >= 1:
        output_upper, output_surface = ort_session_1.run(None, {'input': input_upper, 'input_surface': input_surface})
        hour_diff -= 1
    input_upper = output_upper
    input_surface = output_surface

end_time = time.time()
print("Inference Done!")
print(f"Inference time: {end_time - start_time:.2f} seconds")
# Save the results
np.save(os.path.join(output_data_dir, forecast_time.strftime("%Y-%m-%d") + '_output_upper'), output_upper)
np.save(os.path.join(output_data_dir, forecast_time.strftime("%Y-%m-%d") + '_output_surface'), output_surface)