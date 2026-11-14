"""Render the four manuscript figures with the existing Appendix B scales.

The score-averaging source remains authoritative for the selection algorithm.
Only its calculation functions are loaded, without executing its file-writing
pipeline. Run with the nature-figure scripts directory on PYTHONPATH.
"""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "fig_revisions/output/unified_dispersion_20261016"
MODELS = ["pangu", "fengwu_v1", "fengwu_v2", "fuxi"]
LABELS = ["Pangu", "FengWu v1", "FengWu v2", "FuXi"]
COLORS = ["#1f77b4", "#8BCF8B", "#42949E", "#d62728"]
MARKERS = ["o", "s", "^", "D"]
OFFSETS = [-6, -2, 2, 6]
PARAMETERS = [
    "cape", "cin", "k_idx", "positive_helicity",
    "negative_helicity", "lcl", "lfc", "stp",
]
PARAMETER_LABELS = [
    "CAPE", "CIN", "K Index", "0-3 km Positive SRH",
    "0-3 km Negative SRH", "LCL", "LFC", "STP",
]
HOURS = np.array([24, 48, 72, 120, 168])
CASES = [
    "2013-05-20-20UTC", "2013-05-31-23UTC", "2015-05-07-02UTC",
    "2015-05-25-20UTC", "2016-05-09-21UTC", "2023-04-20-02UTC",
    "2024-04-28-04UTC", "2024-05-07-02UTC", "2024-05-26-04UTC",
    "2025-05-19-23UTC",
]
SOURCES = {}
RECORDS = {}


def load_functions(path, names):
    """Load named functions without executing a legacy script's main pipeline."""
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    assert len(nodes) == len(names)
    namespace = {"np": np, "stats": stats}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def summarize_cases(values):
    """Keep the ten-case arithmetic mean and the unchanged selection rule."""
    values = np.asarray(values, dtype=float)
    assert values.shape == (10,)
    assert np.all(np.isfinite(values))
    functions = load_functions(
        ROOT / "Score_avg.py",
        ["assess_distribution", "check_distribution_type", "calculate_dispersion_metrics"],
    )
    return functions["calculate_dispersion_metrics"](values)


def record_source(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def load_array(path):
    record_source(path)
    return np.load(path)


def save_figure(fig, name):
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=OUTPUT / f"{name}.alignment.json", strict=True,
    )
    fig.savefig(OUTPUT / f"{name}.pdf")
    fig.savefig(OUTPUT / f"{name}.svg")
    fig.savefig(OUTPUT / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_score_grid(metric):
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.2))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.92, hspace=0.48, wspace=0.30)
    handles = []
    rows = []
    for index, parameter in enumerate(PARAMETERS):
        ax = axes.flat[index]
        for model_index, model in enumerate(MODELS):
            if metric == "fss":
                path = ROOT / "avg_scores" / model / f"scores_meanvar_{parameter}_{model}.npz"
                record_source(path)
                with np.load(path, allow_pickle=True) as data:
                    mean = data["mean"][:, 2].copy()
                    dispersion = data["dispersion"][:, 2].copy()
                    methods = data["dispersion_names"][:, 2].astype(str)
            else:
                path = ROOT / "avg_mode" / model / f"mode_summary_{parameter}_{model}.csv"
                record_source(path)
                data = pd.read_csv(path).sort_values("Lead Time (h)")
                np.testing.assert_array_equal(data["Lead Time (h)"], HOURS)
                mean = data["interest_mean"].to_numpy()
                dispersion = data["interest_disp"].to_numpy()
                methods = data["interest_disp_name"].to_numpy()
            assert mean.shape == (5,)
            assert np.all(np.isfinite(mean))
            assert np.all(np.isfinite(dispersion))
            # Clip display arms to score bounds; retain the full scale in source data.
            lower = np.minimum(dispersion, mean)
            upper = np.minimum(dispersion, 1 - mean)
            shift = ScaledTranslation(OFFSETS[model_index] / 72, 0, fig.dpi_scale_trans)
            handle = ax.errorbar(
                HOURS, mean, yerr=np.vstack([lower, upper]), transform=ax.transData + shift,
                color=COLORS[model_index], ecolor=COLORS[model_index],
                marker=MARKERS[model_index], markersize=3.1,
                markeredgecolor="white", markeredgewidth=0.35,
                linewidth=0.85, elinewidth=0.65, capsize=1.0, capthick=0.65,
                label=LABELS[model_index],
            )
            if index == 0:
                handles.append(handle)
            rows.append({
                "model": model, "parameter": parameter, "hours": HOURS.tolist(),
                "mean": mean.tolist(), "dispersion": dispersion.tolist(),
                "method": methods.tolist(), "display_offset_points": OFFSETS[model_index],
            })
        ax.set_xlim(12, 180)
        ax.set_ylim(0, 1)
        ax.set_xticks(HOURS)
        ax.set_title(f"({chr(97 + index)}) {PARAMETER_LABELS[index]}", loc="left", fontsize=9)
        if metric == "fss":
            ax.set_ylabel("FSS")
        else:
            ax.set_ylabel("Total Interest")
        if index >= 6:
            ax.set_xlabel("Lead Time (hours)")
        ax.grid(axis="y", color="0.84", linewidth=0.45)
        ax.set_axisbelow(True)
    fig.legend(handles, LABELS, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=4, fontsize=8)
    fig.canvas.draw()
    # Fixed display widths group offset markers; they are not forecast intervals.
    for ax in axes.flat:
        for hour in HOURS:
            center = ax.transData.transform((hour, 0))[0]
            half_width = 9 * fig.dpi / 72
            left = ax.transData.inverted().transform((center - half_width, 0))[0]
            right = ax.transData.inverted().transform((center + half_width, 0))[0]
            ax.axvspan(left, right, facecolor="0.94", edgecolor="none", zorder=0)
    RECORDS[metric] = rows
    if metric == "fss":
        figure_number = 5
    else:
        figure_number = 6
    save_figure(fig, f"fig{figure_number}_{metric}_appendix_b")


def plot_frequency_bias():
    functions = load_functions(
        ROOT / "fig_revisions/scripts/analyze_168h.py", ["compute_fb"],
    )
    rows = []
    for model in MODELS:
        for parameter in PARAMETERS:
            values = []
            for case in CASES:
                reference = load_array(ROOT / case / f"{parameter}_real.npy")
                forecast = load_array(ROOT / case / model / f"{parameter}_{model}.npy")[4]
                values.append(functions["compute_fb"](forecast, reference, parameter))
            mean, dispersion, method = summarize_cases(values)
            rows.append({
                "model": model, "parameter": parameter, "values": values,
                "n": len(values), "mean": mean, "dispersion": dispersion, "method": method,
            })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.94, hspace=0.52, wspace=0.30)
    bar_colors = ["#e41a1c", "#a65628", "#984ea3", "#ff7f00", "#a6cee3", "#377eb8", "#4daf4a", "#f781bf"]
    tick_labels = ["CAPE", "CIN", "K Index", "SRH+", "SRH-", "LCL", "LFC", "STP"]
    upper = 3.5
    for row in rows:
        upper = max(upper, (row["mean"] + row["dispersion"]) * 1.15)
    for model_index, model in enumerate(MODELS):
        selected = []
        for row in rows:
            if row["model"] == model:
                selected.append(row)
        means = []
        scales = []
        for row in selected:
            means.append(row["mean"])
            scales.append(row["dispersion"])
        ax = axes.flat[model_index]
        x = np.arange(8)
        ax.bar(x, means, color=bar_colors, edgecolor="black", linewidth=0.35, alpha=0.85)
        lower = np.minimum(scales, means)
        ax.errorbar(x, means, yerr=np.vstack([lower, scales]), fmt="none", ecolor="black", capsize=2, linewidth=0.7)
        ax.axhline(1, color="0.45", linestyle="--", linewidth=0.7)
        for index, mean in enumerate(means):
            label_y = mean + scales[index] + upper * 0.025
            if abs(label_y - 1) < upper * 0.04:
                label_y = 1 + upper * 0.04
            ax.text(index, label_y, f"{mean:.2f}", ha="center", fontsize=6.5)
        ax.set_ylim(0, upper)
        ax.set_xticks(x, tick_labels, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Frequency Bias (FB)")
        ax.set_title(f"({chr(97 + model_index)}) {LABELS[model_index]}, 168 h", loc="left", fontsize=9)
        ax.set_axisbelow(True)
    RECORDS["frequency_bias"] = rows
    save_figure(fig, "fig9_fb_168h_appendix_b")


def plot_spatial_sd_ratio():
    rows = []
    for model in ["fengwu_v1", "fengwu_v2"]:
        for lead_index, hour in enumerate(HOURS):
            values = []
            for case in CASES:
                reference = load_array(ROOT / case / "cape_real.npy")
                forecast = load_array(ROOT / case / model / f"cape_{model}.npy")[lead_index]
                reference_sd = np.nanstd(reference)
                assert reference_sd > 0
                values.append(np.nanstd(forecast) / reference_sd)
            mean, dispersion, method = summarize_cases(values)
            rows.append({
                "model": model, "hour": int(hour), "values": values, "n": len(values),
                "mean": mean, "dispersion": dispersion, "method": method,
            })
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.88)
    x = np.arange(5)
    for index, model in enumerate(["fengwu_v1", "fengwu_v2"]):
        means = []
        scales = []
        for row in rows:
            if row["model"] == model:
                means.append(row["mean"])
                scales.append(row["dispersion"])
        positions = x + (index - 0.5) * 0.30
        ax.bar(positions, means, width=0.30, yerr=scales,
               color=COLORS[index + 1], label=LABELS[index + 1],
               edgecolor="white", linewidth=0.5, capsize=3, error_kw={"elinewidth": 1})
        for lead_index, mean in enumerate(means):
            label_y = mean + scales[lead_index] + 0.035
            if abs(label_y - 1) < 0.06:
                label_y = 1.04
            ax.text(positions[lead_index], label_y,
                    f"{mean:.3f}", ha="center", fontsize=8)
    ax.axhline(1, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x, HOURS)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("Spatial SD Ratio (forecast / ERA5)")
    ax.set_axisbelow(True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.99), ncol=2, fontsize=9)
    RECORDS["spatial_sd_ratio"] = rows
    save_figure(fig, "fig12_cape_sd_ratio_appendix_b")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    record_source(ROOT / "Score_avg.py")
    record_source(ROOT / "fig_revisions/scripts/analyze_168h.py")
    record_source(Path(__file__).resolve())
    plot_score_grid("fss")
    plot_score_grid("mode")
    plot_frequency_bias()
    plot_spatial_sd_ratio()
    (OUTPUT / "source_data.json").write_text(json.dumps(RECORDS, indent=2) + "\n")
    (OUTPUT / "source_sha256.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print(f"Saved four figures and source data in {OUTPUT}")


if __name__ == "__main__":
    main()
