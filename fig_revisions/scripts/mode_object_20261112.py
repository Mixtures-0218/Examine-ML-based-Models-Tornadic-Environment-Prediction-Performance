"""Generate the manuscript MODE object-count figure (Fig. 6, 1110 revision).

Six main-text parameters are shown as small multiples: the number of high-value
objects identified in the forecast fields of each parameter, compared with the
number of objects in the ERA5 reference field. A seventh panel shows the
centroid displacement of matched STP objects.

Dispersion bars follow the scale-selection procedure of Appendix B in the
manuscript (Shapiro-Wilk + standardized skewness/kurtosis -> sample SD,
trimmed SD, or robust MAD/IQR scale). Bars are omitted when that procedure
returns a zero scale, which happens for concentrated count distributions.

Outputs: PDF/SVG/PNG plus a JSON of the plotted values under
fig_revisions/output/mode_object_20261112/.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "fig_revisions/output/mode_object_20261112"

TIMES = [
    "2013-05-20-20UTC", "2013-05-31-23UTC", "2015-05-07-02UTC",
    "2015-05-25-20UTC", "2016-05-09-21UTC", "2023-04-20-02UTC",
    "2024-04-28-04UTC", "2024-05-07-02UTC", "2024-05-26-04UTC",
    "2025-05-19-23UTC",
]
MODELS = ["pangu", "fengwu_v1", "fengwu_v2", "fuxi"]
MODEL_LABELS = {"pangu": "Pangu", "fengwu_v1": "FengWu v1",
                "fengwu_v2": "FengWu v2", "fuxi": "FuXi"}
PARAMS = ["cape", "k_idx", "positive_helicity", "lcl", "lfc", "stp"]
PARAM_LABELS = {"cape": "CAPE", "k_idx": "K Index",
                "positive_helicity": "0-3 km Positive SRH",
                "lcl": "LCPL", "lfc": "LFC", "stp": "STP"}
LEADS = [24, 48, 72, 120, 168]
GRID_KM = 25.0
COLORS = {"pangu": "#1f77b4", "fengwu_v1": "#2e8b57",
          "fengwu_v2": "#C87500", "fuxi": "#d62728"}
MARKERS = ["o", "s", "^", "D"]
OFFSETS = [-6, -2, 2, 6]
YMAX_COUNT = 3.3
YMAX_DISP = 200.0


# --------------------------------------------------------------------------
# Appendix B scale-selection procedure
# --------------------------------------------------------------------------
def assess_distribution(data, alpha=0.05, z_general=1.96, z_extreme=3.0):
    """Shapiro-Wilk plus standardized skewness/kurtosis (n = number of events)."""
    values = np.asarray(data, dtype=float).flatten()
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 8:
        return {"dist_type": "insufficient", "skew_type": "unknown",
                "z_kurt": np.nan}

    shapiro_p = stats.shapiro(values)[1]
    skewness = stats.skew(values, bias=False)
    kurtosis = stats.kurtosis(values, fisher=True, bias=False)
    z_skew = skewness / np.sqrt(6.0 / n)
    z_kurt = kurtosis / np.sqrt(24.0 / n)

    if abs(z_skew) <= z_general:
        skew_type = "symmetric"
    elif abs(z_skew) <= z_extreme:
        skew_type = "mild_right" if skewness > 0 else "mild_left"
    else:
        skew_type = "extreme_right" if skewness > 0 else "extreme_left"

    is_normal = (shapiro_p >= alpha and abs(z_skew) <= z_general
                 and abs(z_kurt) <= z_general)
    return {"dist_type": "normal" if is_normal else "non_normal",
            "skew_type": skew_type, "z_kurt": z_kurt}


def selected_dispersion(data):
    """Return (dispersion, method) following the Appendix B branch order."""
    values = np.asarray(data, dtype=float).flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, "no_data"

    info = assess_distribution(values)
    if info["dist_type"] == "normal":
        return float(np.std(values, ddof=1)), "std"

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    filtered = values[(values >= q1 - 3 * iqr) & (values <= q3 + 3 * iqr)]

    if (info["skew_type"] in ["mild_left", "mild_right"]
            and abs(info["z_kurt"]) <= 3.0):
        if len(filtered) > len(values) * 0.9:
            return float(np.std(filtered, ddof=1)), "std_trimmed"
        mad = np.median(np.abs(values - np.median(values)))
        scale = mad * 1.4826 if mad > 0 else iqr / 1.35
        return float(scale), "robust_std"

    mad = np.median(np.abs(values - np.median(values)))
    scale = mad * 1.4826 if mad > 0 else iqr / 1.35
    return float(scale), "robust_mad"


# --------------------------------------------------------------------------
# Load case-level MODE summaries
# --------------------------------------------------------------------------
def load_case_summaries():
    records = []
    for time in TIMES:
        for model in MODELS:
            for param in PARAMS:
                path = os.path.join(time, model, "MODE",
                                    f"MODE_summary_{param}_{model}.csv")
                path = ROOT / path
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                for _, row in frame.iterrows():
                    records.append({
                        "time": time, "model": model, "param": param,
                        "lead": int(row["Lead Time (h)"]),
                        "n_obs": float(row["#Obs Obj"]),
                        "n_fcst": float(row["#Fcst Obj"]),
                        "n_match": float(row["#Matches"]),
                        "cent_grid": float(row["Mean Centroid Dist (grid)"]),
                    })
    return pd.DataFrame(records)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_case_summaries()

    expected_obs = 485.0  # six main parameters x ten events per model
    obs_total = data.groupby("model").n_obs.sum()
    assert np.allclose(obs_total.values, expected_obs), obs_total

    # ---- per parameter: ERA5 object count (independent of lead time) ------
    era5 = {}
    for param in PARAMS:
        subset = data[(data.model == "pangu") & (data.param == param)]
        per_event = subset.drop_duplicates("time").set_index("time").n_obs
        mean_value = float(per_event.mean())
        scale, method = selected_dispersion(per_event.values)
        era5[param] = {"mean": mean_value, "dispersion": scale,
                       "method": method, "per_event": per_event.to_dict()}

    # ---- per parameter/model/lead: forecast object count -----------------
    counts = {}
    for param in PARAMS:
        counts[param] = {}
        for model in MODELS:
            counts[param][model] = {}
            for lead in LEADS:
                subset = data[(data.model == model) & (data.param == param)
                              & (data.lead == lead)]
                per_event = subset.set_index("time").n_fcst
                scale, method = selected_dispersion(per_event.values)
                counts[param][model][lead] = {
                    "mean": float(per_event.mean()),
                    "dispersion": scale, "method": method,
                    "n_events": int(len(per_event)),
                }

    # ---- STP matched-object centroid displacement ------------------------
    displacement = {}
    for model in MODELS:
        displacement[model] = {}
        for lead in LEADS:
            subset = data[(data.model == model) & (data.param == "stp")
                          & (data.lead == lead) & (data.n_match > 0)]
            values = subset.set_index("time").cent_grid.values * GRID_KM
            scale, method = selected_dispersion(values)
            displacement[model][lead] = {
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "dispersion": scale, "method": method,
                "n_events": int(len(values)),
            }

    # ---- plot ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    from matplotlib.gridspec import GridSpec
    figure = plt.figure(figsize=(7.2, 5.75), dpi=300)
    grid = GridSpec(3, 3, figure=figure, height_ratios=[1, 1, 0.95],
                    hspace=0.72, wspace=0.28,
                    left=0.085, right=0.99, bottom=0.075, top=0.895)
    panels = [figure.add_subplot(grid[i // 3, i % 3]) for i in range(6)]
    axis_disp = figure.add_subplot(grid[2, :])

    handles = []
    for index, (param, axis) in enumerate(zip(PARAMS, panels)):
        reference = era5[param]
        lower = max(0.0, reference["mean"] - reference["dispersion"])
        upper = min(YMAX_COUNT, reference["mean"] + reference["dispersion"])
        axis.axhspan(lower, upper, facecolor="0.45", alpha=0.16,
                     edgecolor="0.45", linewidth=0.6, linestyle="--", zorder=1.2)
        line, = axis.plot([12, 180], [reference["mean"]] * 2,
                          color="0.35", linewidth=1.1, linestyle="--", zorder=3)
        if index == 0:
            handles.append(line)

        for model_index, model in enumerate(MODELS):
            means, lower_arm, upper_arm = [], [], []
            for lead in LEADS:
                entry = counts[param][model][lead]
                means.append(entry["mean"])
                lower_arm.append(min(entry["dispersion"], entry["mean"]))
                upper_arm.append(min(entry["dispersion"],
                                     YMAX_COUNT - entry["mean"]))
            shift = ScaledTranslation(OFFSETS[model_index] / 72, 0,
                                      figure.dpi_scale_trans)
            handle = axis.errorbar(
                LEADS, means, yerr=np.vstack([lower_arm, upper_arm]),
                transform=axis.transData + shift, color=COLORS[model],
                ecolor=COLORS[model], marker=MARKERS[model_index],
                markersize=2.9, markeredgecolor="white",
                markeredgewidth=0.3, linewidth=0.8, elinewidth=0.6,
                capsize=0.9, capthick=0.6)
            if index == 0:
                handles.append(handle)

        axis.set_xlim(12, 180)
        axis.set_ylim(0, YMAX_COUNT)
        axis.set_xticks(LEADS)
        axis.set_title(f"({chr(97 + index)}) {PARAM_LABELS[param]}",
                       loc="left", fontsize=8)
        axis.grid(axis="y", color="0.85", linewidth=0.4)
        axis.set_axisbelow(True)
        if index % 3 == 0:
            axis.set_ylabel("Objects per case")
        if index >= 3:
            axis.set_xlabel("Lead Time (hours)")

    for model_index, model in enumerate(MODELS):
        means, lower_arm, upper_arm = [], [], []
        for lead in LEADS:
            entry = displacement[model][lead]
            means.append(entry["mean"])
            if np.isfinite(entry["dispersion"]) and np.isfinite(entry["mean"]):
                lower_arm.append(min(entry["dispersion"], entry["mean"]))
                upper_arm.append(min(entry["dispersion"],
                                     YMAX_DISP - entry["mean"]))
            else:
                lower_arm.append(np.nan)
                upper_arm.append(np.nan)
        shift = ScaledTranslation(OFFSETS[model_index] / 72, 0,
                                  figure.dpi_scale_trans)
        axis_disp.errorbar(
            LEADS, means, yerr=np.vstack([lower_arm, upper_arm]),
            transform=axis_disp.transData + shift, color=COLORS[model],
            ecolor=COLORS[model], marker=MARKERS[model_index],
            markersize=3.2, markeredgecolor="white", markeredgewidth=0.3,
            linewidth=0.9, elinewidth=0.7, capsize=1.0, capthick=0.7,
            label=MODEL_LABELS[model])
    axis_disp.set_xlim(12, 180)
    axis_disp.set_ylim(0, YMAX_DISP)
    axis_disp.set_xticks(LEADS)
    axis_disp.set_xlabel("Lead Time (hours)")
    axis_disp.set_ylabel("STP object centroid\ndisplacement (km)")
    axis_disp.set_title("(g) Displacement of matched STP objects",
                        loc="left", fontsize=8)
    axis_disp.grid(axis="y", color="0.85", linewidth=0.4)
    axis_disp.set_axisbelow(True)
    axis_disp.text(
        0.005, 0.04,
        "open-marker leads: fewer than 8 events contain a matched STP object "
        "(no Appendix B error bar)",
        transform=axis_disp.transAxes, fontsize=5.6, color="0.35")

    figure.canvas.draw()
    for axis in panels + [axis_disp]:
        for hour in LEADS:
            center = axis.transData.transform((hour, 0))[0]
            half_width = 9 * figure.dpi / 72
            left = axis.transData.inverted().transform((center - half_width, 0))[0]
            right = axis.transData.inverted().transform((center + half_width, 0))[0]
            axis.axvspan(left, right, facecolor="0.94", edgecolor="none",
                         zorder=0)

    figure.legend(handles, ["ERA5 (observed)"] + [MODEL_LABELS[m] for m in MODELS],
                  loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=5,
                  fontsize=7)

    stem = OUTPUT / "fig6_mode_objects"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    payload = {"era5": era5, "forecast_counts": counts,
               "stp_displacement": displacement,
               "times": TIMES, "parameters": PARAMS, "models": MODELS,
               "leads": LEADS}
    (OUTPUT / "plotted_values.json").write_text(json.dumps(payload, indent=2))
    print("Saved", stem.with_suffix(".pdf"))
    print("Saved", stem.with_suffix(".svg"))
    print("Saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
