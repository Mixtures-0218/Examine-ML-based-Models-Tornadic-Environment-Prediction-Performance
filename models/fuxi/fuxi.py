# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import argparse
import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort

from util import save_like

# Silence ONNX Runtime logs
ort.set_default_logger_severity(3)

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="FuXi ONNX model directory")
parser.add_argument('--input', type=str, required=True, help="Input NetCDF file")
parser.add_argument('--save_dir', type=str, default="", help="Directory to save final output")
parser.add_argument('--num_steps', type=int, nargs="+", default=[20], help="Steps for short/medium/long")
args = parser.parse_args()

# ============================================================
# Time embedding (FuXi standard, 6-hourly)
# ============================================================
def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []

    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, "H") for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]

        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)

        tembs.append(temb)

    return np.stack(tembs)

# ============================================================
# Load ONNX model (GPU)
# ============================================================
def load_model(model_path):
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1

    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested"
    }

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=[("CUDAExecutionProvider", cuda_provider_options)]
    )
    return session

# ============================================================
# Main inference logic
# ============================================================
def run_inference(model_dir, data, num_steps, save_dir):

    # ---- Clean output directory ----
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for f in os.listdir(save_dir):
            if f.endswith(".nc"):
                os.remove(os.path.join(save_dir, f))

    total_step = sum(num_steps)
    init_time = pd.to_datetime(data.time.values[-1])
    tembs = time_encoding(init_time, total_step)

    print(f"Init time : {init_time.strftime('%Y%m%d-%H')}")
    print(f"Steps     : {total_step}")

    # ---- Ensure latitude order (90 → -90) ----
    if data.lat.values[0] < data.lat.values[-1]:
        data = data.sortby("lat", ascending=False)

    assert data.lat.values[-1] == -90

    # ---- Model input ----
    input_tensor = data.values[None]
    print(f"Input shape: {input_tensor.shape}")

    stages = ["short", "medium", "long"]
    step = 0

    for i, nstep in enumerate(num_steps):
        stage = stages[i]
        model_path = os.path.join(model_dir, f"{stage}.onnx")

        print(f"\nLoading {stage} model...")
        session = load_model(model_path)

        print(f"Running {stage} inference...")
        start = time.perf_counter()

        for _ in range(nstep):
            temb = tembs[step]
            new_input, = session.run(None, {
                "input": input_tensor,
                "temb": temb
            })

            output = new_input[:, -1]

            # ✅ ONLY save the final step
            if step == total_step - 1:
                save_like(output, data, step, save_dir)
                print("✅ Final step saved")

            print(
                f"stage={stage}, step={step+1:02d}, "
                f"range={output.min():.2f} ~ {output.max():.2f}"
            )

            input_tensor = new_input
            step += 1

        print(f"{stage} stage finished in {time.perf_counter() - start:.2f}s")

        if step >= total_step:
            break

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    data = xr.open_dataarray(args.input)
    run_inference(args.model, data, args.num_steps, args.save_dir)

# -


