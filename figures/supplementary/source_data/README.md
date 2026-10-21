# Numerical source data

These tables export the saved values used in the complete eight-parameter figures. They do not recompute scores or dispersion estimates. NaN retains a missing saved value without imputation.

- `performance_diagram.csv`: 160 model/parameter/lead-time rows; mean POD, CSI, FSS and SR, with the corresponding selected cross-event dispersion scales. The performance diagrams themselves do not display these dispersion scales.
- `fss.csv`, `mode_total_interest.csv`: saved means, selected cross-event dispersion scales and selection-method identifiers for S3/S4.
- `frequency_bias.csv`: 168 h cross-event frequency-bias values for S5.
- `case_frequency_bias.csv`: 168 h case frequency-bias values for S6.
- `case_mode_total_interest.csv`: case-level mean MODE total interest for S7.
- `plotted_data.json`: the original plotting record; it also includes the six-parameter main displays.

The ZIP input cache retains the original NPZ/CSV/JSON bytes and is the authoritative input for rerendering. CSV exports are an additional inspection format. Error bars summarize variability across ten events, not standard errors or confidence intervals for the mean. The case involving GFS is separate and is not included in cross-event averages.
