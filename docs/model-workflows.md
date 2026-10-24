# Model workflows

The model and download scripts coexist on `main`; use the component directory as
the working directory. Their relative imports, input names, case settings, and
scientific calculations are unchanged. The original branch READMEs are retained
verbatim; the working-directory instructions and clarifications below accompany them.

## Component dependencies

Use an environment compatible with your model weights and GPU runtime. This migration
does not select new package versions or unify the separate model environments.

| Component | Python imports used by its scripts |
|---|---|
| FengWu | cdsapi, numpy, netCDF4, onnx, onnxruntime with CUDA support |
| FuXi | cdsapi, numpy, pandas, xarray, onnxruntime with CUDA support; cartopy and matplotlib for the plotting helper |
| Pangu-Weather | cdsapi, numpy, netCDF4, xarray, onnx, onnxruntime with CUDA support |
| GFS primary downloader / converter | requests, numpy, pygrib, netCDF4 |
| GFS alternate downloader | herbie-data (provides the herbie import) |

NetCDF reading and writing also require a compatible backend. ERA5 downloads require
local CDS API authentication. Model weights and downloaded meteorological data are
external inputs, not bundled in this repository.

## FengWu V1 / V2

The original entry point is `output_decode.py`, with `models/fengwu/` as its
working directory. This preserved snapshot is not a runnable end-to-end workflow:

- `inference.py` imports `init_time_list`, but `era5_data_get.py` defines
  `init_time` instead. The import cannot complete successfully.
- The downloader saves `input1.npy` and `input2.npy`, while inference reads
  lead-specific filenames such as `input1_24.npy` and `input2_24.npy`.

Resolve these existing interfaces in a separate change before attempting the full
FengWu workflow. They are documented here without altering the historical scripts.

Place `fengwu_v1.onnx` and `fengwu_v2.onnx` in this directory. The bundled
`data_mean.npy` and `data_std.npy` are required normalization resources and remain
version controlled.

Before running, inspect the case dates in `era5_data_get.py` and `inference.py`.
The legacy README's `inferece.py` refers to `inference.py`.
The decoder imports `inference.py`, which imports the ERA5 downloader. Imports
execute top-level code: starting the decoder can trigger downloads before reaching
the import failure described above. With the interfaces reconciled, this entry
point also runs inference before decoding.
Do not run inference separately and then assume that starting the decoder skips it.
Inputs and outputs resolve under this working directory in `input_data/` and
`output_data/`.

## FuXi

From the repository root:

```bash
cd models/fuxi
python era5_get_new.py
python build_inputs.py
python fuxi.py --model FuXi_EC --input inputs/input_20230420_02_lead024h.nc --save_dir outputs/
```

The input filename above is the original branch README's example; use the file
generated for your selected case. Inspect the case and path settings in the download
and input-building scripts before running. Place the required ONNX weights under
`FuXi_EC/`, or pass another local model directory with `--model`.

The original `--num_steps` default is `[20]`, corresponding to 120 hours in
six-hour steps using the short stage. Select stages and steps for the intended run.
The inference script deletes existing `.nc` files in the directory passed to
`--save_dir`; use a dedicated output directory for each run.

## Pangu-Weather

From the repository root:

```bash
cd models/pangu
mkdir -p output_data
python inference_gpu.py
python output_decode.py
```

Place the four required weights (`pangu_weather_1.onnx`,
`pangu_weather_3.onnx`, `pangu_weather_6.onnx`, and
`pangu_weather_24.onnx`) in this directory.
The inference script imports the ERA5 downloader, asks for initialization and forecast
times, and uses local `input_data/` and `output_data/` directories.
The decoder writes fixed output names; keep separate runs in separate output
directories to preserve earlier decoded files.

## GFS download and conversion

From the repository root:

```bash
cd data_download/gfs
python gfs_data_get.py
```

The primary script downloads forecasts and performs conversion. The separate
`convert_GRIB2_NC.py` retains its original machine-specific `grib_dir`;
set it to the intended local directory before using that standalone converter.
`run_gfs_download.py` is a separate Herbie-based downloader; inspect its case
settings before use.

For data access that requires RDA authentication, create your own local
`rdams_token.txt` in this working directory, following the retained component
README. This filename is ignored by Git. The credential file tracked on the old
GFS branch was deliberately excluded from the migration. The old branch and its
history are retained; this migration does not remove previously committed credentials
from history. If that credential is still active, its owner should revoke and replace it.

The original `GFS Set Up_Converting Data.docx` is preserved unchanged.

## Handoff to evaluation

The component scripts retain their original output conventions. They do not
automatically assemble the evaluation directory structure. Prepare the resulting
NetCDF fields in the event/model layout described in the
[root README](../README.md#1-compute-tornado-relevant-parameters), then run the
existing root-level parameter and verification scripts from the repository root.

Existing evaluation scripts, manuscript figures, supplementary assets, thresholds,
and model settings are unchanged. The `FSS_neighborhood_size_verify` branch is
outside this migration and remains at its original commit.

## Verification scope

Migration integrity is checked against the source Git blob IDs in
[model-code-migration.json](model-code-migration.json). Syntax checks do not import
or execute the scripts. Full downloads and GPU inference require external credentials,
model weights, input data, and compatible runtime environments; they are not established
by a directory-only migration.
