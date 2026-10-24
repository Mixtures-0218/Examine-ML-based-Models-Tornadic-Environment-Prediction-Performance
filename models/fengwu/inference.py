import os
import numpy as np
import onnx
import onnxruntime as ort
import time
from datetime import datetime
from era5_data_get import init_time_list

models = ['fengwu_v1', 'fengwu_v2']
for model in models:
    for init_time in init_time_list:
        # Forecast time input
        print("Please input the forecast time, the import time hour should be divisible by 6!")
        flag = 0
        while flag==0:
            forecast_time = datetime(
                year=2015,
                month=5,
                day=7,
                hour=2,
                minute=0)
            '''
            forecast_time = datetime(
                year=int(input("year: ")),
                month=int(input("month: ")),
                day=int(input("day: ")),
                hour=int(input("hour: ")),
                minute=0)
            '''
            # Calculate the time difference
            hour_diff = (forecast_time - init_time).total_seconds() / 3600
            if hour_diff % 6 != 0:
                print("The import time hour should be divisible by 6!")
            else:
                loop = int(hour_diff / 6)
                flag = 1
        
        
        # The directory of your input and output data
        input_data_dir = os.path.join(
            os.path.join(os.getcwd(), "input_data"),
        )
        output_data_dir = os.path.join(
            os.path.join(os.getcwd(), "output_data"),
        )
        os.makedirs(output_data_dir, exist_ok=True)
        
        # Loading model
        model_6 = onnx.load(f'{model}.onnx')
        
        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = 4
        
        # Set the behavier of cuda provider
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}
        
        # Initialize onnxruntime session for Models
        ort_session_6 = ort.InferenceSession(f'{model}.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
        
        data_mean = np.load("data_mean.npy")[:, np.newaxis, np.newaxis]
        data_std = np.load("data_std.npy")[:, np.newaxis, np.newaxis]
        
        input1 = np.load(os.path.join(input_data_dir, f'input1_{int(hour_diff)}.npy')).astype(np.float32)
        input2 = np.load(os.path.join(input_data_dir, f'input2_{int(hour_diff)}.npy')).astype(np.float32)
        
        # Normalization method
        input1_after_norm = (input1 - data_mean) / data_std
        input2_after_norm = (input2 - data_mean) / data_std
        
        # Merge the two input file into one
        input = np.concatenate((input1_after_norm, input2_after_norm), axis=0)[np.newaxis, :, :, :]
        input = input.astype(np.float32)
        
        # Inference logic
        start_time = time.time()
        print("Start to inference...")
        for i in range(loop):
            output = ort_session_6.run(None, {'input':input})[0]
            input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)
            # Anti-normalization
            output = (output[0, :69] * data_std) + data_mean
            print(output.shape)
            i += 1
            print("Inference:", i)
        np.save(os.path.join(output_data_dir, f"output_{model}_{int(hour_diff)}.npy"), output)
        end_time = time.time()
        print("Inference Done!")
        print(f"Inference time: {end_time - start_time:.2f} seconds")