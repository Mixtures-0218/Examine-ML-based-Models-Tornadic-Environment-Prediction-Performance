# Examine-ML-based-Models-Tornadic-Environment-Prediction-Performance

Code Contributors: Yiding Luo, Rui Yuan, Shetian Xia, and Royce Lee

All the necessary code for

> **Evaluation of Machine-Learning Weather Models in Reproducing Mesoscale Tornado-Favorable Convective Environments**
> (Yiding Luo, Rui Yuan, Shetian Xia, Royce Lee, and Alexandra Anderson-Frey; arXiv preprint, 2026)

The repository contains the full analysis pipeline (model output data → tornado-relevant
parameters → verification metrics → figures) together with the final manuscript figures.

## 1. Compute tornado-relevant parameters

Run `tornado_parameters.py` to get the parameters (CAPE, K-Index, SRH, LCL, LFC, CIN, etc.) once
all the necessary model output data have been obtained. Change the variables `models` and `times`
to control which model and which event time to compute.

The result folder should be named `YYYY-MM-DD-HHUTC` under the same path as
`tornado_parameters.py`, with the structure

```plain
├── root
│   ├── YYYY-MM-DD-HHUTC
│       ├── real.nc
│       ├── real_surface.nc
│       ├── fengwu_v1
│           ├── fengwu_v1_24.nc
│           ├── fengwu_v1_48.nc
│           ├── fengwu_v1_72.nc
│           ├── fengwu_v1_120.nc
│           ├── fengwu_v1_168.nc
│       ├── fengwu_v2
│       ├── pangu
│       ├── fuxi
```

The other models' results should follow the same structure as `fengwu_v1` above. The `.nc` file
naming format is `ModelName_LeadTime.nc`. The parameter output files are stored in the
corresponding `YYYY-MM-DD-HHUTC` folder.

Run `STP.py` after computing the above parameters to get the significant tornado parameter (STP)
data in the corresponding `YYYY-MM-DD-HHUTC` folder.

## 2. Verification metrics (POD, SR, CSI, FB, FSS, ...)

Run `Scores.py` with the intended variables `models`, `times`, and `parameters`. The results are
stored in each `YYYY-MM-DD-HHUTC` folder under a subfolder named `scores`.

`Score_avg.py` averages the scores over the selected times and computes the spread; the results are
several `.csv` files stored in the `avg_scores` folder.

## 3. Performance diagram

- `performance_diagram.py` — single target time, single model, single parameter, multiple lead
  times (variables: `time`, `model`, `parameter`).
- `performance_diagram_avg.py` — single parameter and model with averaged target times and
  multiple lead times (variables: `model`, `parameter`; requires the `avg_scores` folder).
- `performance_diagram_cross_avg.py` — multiple models, parameters, and lead times with averaged
  target times; the function parameter `draw_ellipse` controls the 1σ spread.

## 4. Fractions Skill Score (FSS)

Run `FSS_avg_plot.py` to plot scores for multiple parameters and models with averaged target
times. The output figures are stored in the `avg_fss` folder.

## 5. Method for Object-based Diagnostic Evaluation (MODE)

- `MODE.py` — compute MODE interest values (variables: `models`, `times`, `parameters`); results
  go to the `MODE` subfolder of each `YYYY-MM-DD-HHUTC` folder.
- `MODE_avg.py` — average over the selected times and compute the spread (`avg_scores` folder).
- `MODE_plot.py` — MODE figures for multiple parameters/lead times/models with averaged times.
- `MODE_plot_single.py` — MODE figures for a single target time.

## 6. Visualizations (map fields)

- `visualization.py` — single parameter, model, lead time, and target time on a map.
- `visualization_all.py` — single parameter, lead time, and target time, multiple models.
- `visualization_leads.py` — single parameter, model, and target time, multiple lead times.

## 7. Figure revisions (reviewer round)

`fig_revisions/` contains the revision scripts that regenerate the final manuscript figures
(4 × 2 panel layouts, enlarged fonts, colorbar units, and removal of in-figure titles per the
reviewer comments):

- `FSS_avg_plot.py`, `MODE_plot.py`, `performance_diagram_cross_avg.py` — updated combined
  4 × 2 figures (no duplicated in-figure titles; MODE title typo fixed).
- `rebuild_analysis_figs.py`, `rebuild_fig12_fig14.py`, `plot_figures.py`, `analyze_168h.py`,
  `fig_2025_case_fb.py`, `MODE_avg.py`, `visualization_leads.py` — supporting revision figures
  (vertical profiles, θe–CAPE–LFC relation, 168-h frequency bias, 2025-05-19 case study, etc.).

## 8. Manuscript figures

`figures/` contains the final PNG figures used in the manuscript, numbered as in the paper
(`fig1_oklahoma_study_area` … `fig15_mode_case`; `fig2_data_flow` is the model-inference
flowchart regenerated from `Flowchart.pptx`). The first three figures rely on ERA5 reanalysis
(Copernicus CDS), the GFS archive (NCEI/RDA, doi:10.5065/D65D8PWK), NASA SRTM, and the NOAA
Storm Events database.

## Dependencies

Python 3.9+, `metpy`, `xarray`, `netCDF4`, `numpy`, `pandas`, `scipy`, `matplotlib`, and the MET
(Model Evaluation Tools) package for the MODE interest calculations.
