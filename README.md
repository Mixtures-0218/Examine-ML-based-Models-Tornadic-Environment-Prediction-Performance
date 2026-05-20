# FuXi-Sky

Download ERA5 data, build inputs, and run FuXi model inference.

## End-to-End Workflow

Run these 3 scripts in order:

### 1. Download ERA5 data

```bash
python era5_get_new.py
```

Downloads pressure-level, surface, and daily precipitation data via CDS API.
Target forecast time: `2023-04-20 02:00`, leads: 24/48/72/120/168h.
Files saved to `ERA5{YYYYMMDD}/` directories.

### 2. Build model inputs

```bash
python build_inputs.py
```

Reads the downloaded NetCDF files and assembles FuXi input tensors
of shape `(time=2, level=70, lat=721, lon=1440)`. Outputs to `inputs/`.

### 3. Run inference

```bash
python fuxi.py \
  --model FuXi_EC \
  --input inputs/input_20230420_02_lead024h.nc \
  --save_dir outputs/
```

Arguments:
- `--model`: directory with `short.onnx`, `medium.onnx`, `long.onnx`
- `--input`: input file from step 2
- `--save_dir`: output directory (optional)
- `--num_steps`: steps per stage, default `[20]` (short only, 120h)

## Files

| File | Role |
|---|---|
| `era5_get_new.py` | Download ERA5 data |
| `build_inputs.py` | Build model input tensors |
| `fuxi.py` | Run FuXi inference |
| `util.py` | Helpers for fuxi.py |
| `FuXi_EC/` | ONNX model weights |
| `test.py` | Check inputs for NaN |
