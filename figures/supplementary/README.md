# Complete eight-parameter results

The current manuscript displays six parameters in its repeated main comparisons: CAPE, K Index, positive 0–3 km SRH, LCL, LFC and STP. This collection retains all eight parameters, including CIN and negative 0–3 km SRH. It replaces the long Appendix C figure collection in the working manuscript.

The figures use the same event selection, scores, averaging procedure, selected dispersion scales and model colors as the main analysis. S1 and S2 jointly contain the complete eight-parameter performance diagram; each remaining figure contains all eight parameters.

| Figure | PDF | Content |
|---|---|---|
| S1 | [S1_performance_part1.pdf](S1_performance_part1.pdf) | Performance diagrams: CAPE, CIN, K Index and positive SRH |
| S2 | [S2_performance_part2.pdf](S2_performance_part2.pdf) | Performance diagrams: negative SRH, LCL, LFC and STP |
| S3 | [S3_fss.pdf](S3_fss.pdf) | Eight-parameter FSS across ten events |
| S4 | [S4_mode.pdf](S4_mode.pdf) | Eight-parameter MODE total interest across ten events |
| S5 | [S5_frequency_bias.pdf](S5_frequency_bias.pdf) | Eight-parameter frequency bias at 168 h across ten events |
| S6 | [S6_case_frequency_bias.pdf](S6_case_frequency_bias.pdf) | Eight-parameter frequency bias at 168 h for the 19 May 2025 case |
| S7 | [S7_case_mode.pdf](S7_case_mode.pdf) | Eight-parameter MODE total interest for the 19 May 2025 case |

## CIN and negative SRH

CIN scores were low and fluctuated across lead times rather than following a consistent decline (S1b). The manuscript briefly discusses CIN; its distribution was non-uniform and values were sometimes absent across much of the domain. Negative SRH was rarely observed in the selected environments. Its limited occurrence restricts the representativeness of a model ranking for that parameter (S2e).

## Figure conventions

S1: (a) CAPE, (b) CIN, (c) K Index and (d) positive 0–3 km SRH. S2: (e) negative 0–3 km SRH, (f) LCL, (g) LFC and (h) STP. Curved contours denote CSI and straight reference lines denote frequency bias. Colors/markers identify model versions, and labels/leader lines identify 24, 48, 72, 120 and 168 h. The ERA5 P25 threshold is used for CIN and negative SRH; P75 is used for the other parameters.

S3/S4 show arithmetic means across ten events. Error bars extend by one selected cross-event dispersion scale on either side, clipped to 0–1. Selection among sample SD, trimmed SD and robust MAD/IQR scales follows Appendix B of the manuscript. These bars describe cross-event variability, not uncertainty in the mean. Horizontal marker offsets and pale gray bands are display groups. MODE total interest and FSS describe different properties and are not directly comparable in magnitude. FSS uses a 3×3 grid neighborhood.

S5 shows arithmetic mean frequency bias at 168 h across ten events; error bars use the same dispersion procedure, with negative portions clipped. Parameter colors are separate from model colors. The dashed reference denotes FB = 1. S5 and S6 use the reference P25 lower tail for CIN and negative SRH and P75 for the other parameters.

S6/S7 concern the 19 May 2025 case. Model colors are Pangu blue, FengWu V1 green, FengWu V2 orange, FuXi red and GFS purple. ML forecasts use ERA5 as the reference; GFS uses its reference forecast. S6 has a dashed FB = 1 line; blue/red background bands mark FB below 0.5/above 1.5, while numeric labels use blue below 0.5 and red above 1.2. These are display guides. S7 shows saved case-level mean total interest at each lead time, without cross-event averaging or error bars. GFS reference/lead-time alignment follows the manuscript's case-study limitations.

## Data and rerendering

[Numerical tables](source_data/README.md), [byte-preserved aggregate input cache](plot_inputs.zip), and [input SHA-256 checksums](input_sha256.json) accompany the figures. The cache contains 106 derived-input files; it excludes raw model/reanalysis fields. Of these, 104 retain the previous bytes and two JSON caches contain the corrected negative-SRH FB data and provenance.

From the repository root, using Python with NumPy, pandas and Matplotlib:

```bash
python -m zipfile -e figures/supplementary/plot_inputs.zip .
python fig_revisions/scripts/parameter_focus_appendix_20261020.py
```

Outputs are written to `fig_revisions/output/LFC_focus_appendix_20261020/`. The archived script retains its date and original output names: `figC1`–`figC7` correspond to S1–S7. Six-parameter outputs correspond to current main Figures 4–7 and 12–13. The other main figures are supplied as saved manuscript assets.

The repository copy of the plotting script omits only the import/call to a local external panel-alignment audit utility. The negative-SRH FB arrays in S5/S6 were corrected on 21 October 2026. Other scientific arrays, plot settings, fonts and model colors are unchanged. Original exports underwent alignment and visual review before archiving. Rerendering from this cache reproduces the plotted aggregate results; it does not rerun model inference, parameter calculation or event-level scoring.

## Negative-SRH threshold limitation and correction

The previous S5/S6 FB helpers used >= P75 for negative SRH. All ten ERA5 P75 thresholds were zero, so those values measured zero-region coverage and could not establish overprediction of strong negative SRH. The corrected S5 means, using <= P25, are Pangu 0.638636, FengWu V1 0.534848, FengWu V2 0.581818 and FuXi 0.481818. The original selected-dispersion procedure is retained. S6 uses the same tail rule; GFS is compared with its own reference field.

For 7 May 2015 and 20 April 2023, the ERA5 P25 is also zero. The inclusive event mask then covers the entire non-positive field; these cases are non-discriminating for strong-negative-SRH prediction. No events were removed and no threshold rule was changed. The limitation also applies to interpretation of those event scores in S1-S4. See the [before/after correction record](negative_srh_threshold_correction.json).
