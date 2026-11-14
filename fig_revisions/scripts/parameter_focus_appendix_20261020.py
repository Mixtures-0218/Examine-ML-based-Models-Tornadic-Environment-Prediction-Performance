"""Render six-parameter main figures and complete eight-parameter appendix figures."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

import performance_diagram_consistent_colors_20261019 as performance

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "fig_revisions/output/LFC_focus_appendix_20261020"
MAIN_PARAMETERS = ["cape", "k_idx", "positive_helicity", "lcl", "lfc", "stp"]
ALL_PARAMETERS = ["cape", "cin", "k_idx", "positive_helicity", "negative_helicity", "lcl", "lfc", "stp"]
LABELS = {"cape": "CAPE", "cin": "CIN", "k_idx": "K Index", "positive_helicity": "0-3 km Positive SRH", "negative_helicity": "0-3 km Negative SRH", "lcl": "LCPL", "lfc": "LFCP", "stp": "STP"}
SHORT_LABELS = {"cape": "CAPE", "cin": "CIN", "k_idx": "K Index", "positive_helicity": "SRH+", "negative_helicity": "SRH-", "lcl": "LCPL", "lfc": "LFCP", "stp": "STP"}
MODELS = ["pangu", "fengwu_v1", "fengwu_v2", "fuxi"]
MODEL_LABELS = {"pangu": "Pangu", "fengwu_v1": "FengWu v1", "fengwu_v2": "FengWu v2", "fuxi": "FuXi", "gfs": "GFS"}
COLORS = {"pangu": "#1f77b4", "fengwu_v1": "#2e8b57", "fengwu_v2": "#C87500", "fuxi": "#d62728", "gfs": "#9B59B6"}
MARKERS = ["o", "s", "^", "D"]
OFFSETS = [-6, -2, 2, 6]
PARAMETER_COLORS = {"cape": "#e41a1c", "cin": "#a65628", "k_idx": "#984ea3", "positive_helicity": "#ff7f00", "negative_helicity": "#a6cee3", "lcl": "#377eb8", "lfc": "#4daf4a", "stp": "#f781bf"}
HOURS = np.array([24, 48, 72, 120, 168])
SOURCES = {}
RECORDS = {}


def record_source(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    record_source(path)
    return json.loads(path.read_text())


def save_figure(figure, name):
    figure.canvas.draw()
    figure.savefig(OUTPUT / f"{name}.pdf")
    figure.savefig(OUTPUT / f"{name}.svg")
    figure.savefig(OUTPUT / f"{name}.png", dpi=300)
    plt.close(figure)
    print("Saved", name, flush=True)


def plot_performance(parameters, name, first_letter=0):
    rows = len(parameters) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(14.67, rows * 6.10), dpi=300)
    figure.subplots_adjust(left=0.08, right=0.94, bottom=0.065, top=0.92, hspace=0.48, wspace=0.40)
    panels = []
    for index, parameter in enumerate(parameters):
        axis = axes.flat[index]
        model_data = {}
        for model in MODELS:
            path = ROOT / "avg_scores" / model / f"scores_meanvar_{parameter}_{model}.npz"
            record_source(path)
            with np.load(path) as data:
                means = data["mean"].copy()
                dispersion = data["dispersion"].copy()
            assert means.shape == (5, 4)
            assert np.isfinite(means).all()
            model_data[model] = {"lead_times": ["24h", "48h", "72h", "120h", "168h"], "mean_scores": means, "std_scores": dispersion}
        title = f"({chr(97 + first_letter + index)}) {LABELS[parameter]}"
        performance._draw_performance_diagram(axis, model_data, title, show_legend=True, show_title=False)
        axis.set_title(title, fontsize=20, fontweight="bold", pad=28)
        panels.append((axis, parameter, title, model_data))
    figure.tight_layout(h_pad=4, w_pad=5)
    figure.canvas.draw()
    count = 0
    for axis, parameter, title, model_data in panels:
        original_index = ALL_PARAMETERS.index(parameter)
        original_title = f"({chr(97 + original_index)}) {LABELS[parameter]}"
        axis.set_title(original_title, fontsize=20, fontweight="bold", pad=28)
        annotations = performance._place_lead_time_labels(axis, model_data)
        count += len(annotations)
        axis.set_title(title, fontsize=20, fontweight="bold", pad=28)
    assert count == len(parameters) * 20
    RECORDS[name] = {"parameters": parameters, "lead_time_labels": count}
    save_figure(figure, name)


def plot_score_grid(metric, parameters, name):
    rows = len(parameters) // 2
    height = 7.6
    if rows == 4:
        height = 9.2
    figure, axes = plt.subplots(rows, 2, figsize=(7.2, height))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.92, hspace=0.48, wspace=0.30)
    handles = []
    records = []
    for index, parameter in enumerate(parameters):
        axis = axes.flat[index]
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
            assert np.isfinite(mean).all()
            assert np.isfinite(dispersion).all()
            lower = np.minimum(dispersion, mean)
            upper = np.minimum(dispersion, 1 - mean)
            shift = ScaledTranslation(OFFSETS[model_index] / 72, 0, figure.dpi_scale_trans)
            handle = axis.errorbar(HOURS, mean, yerr=np.vstack([lower, upper]), transform=axis.transData + shift, color=COLORS[model], ecolor=COLORS[model], marker=MARKERS[model_index], markersize=3.1, markeredgecolor="white", markeredgewidth=0.35, linewidth=0.85, elinewidth=0.65, capsize=1.0, capthick=0.65, label=MODEL_LABELS[model])
            if index == 0:
                handles.append(handle)
            records.append({"model": model, "parameter": parameter, "mean": mean.tolist(), "dispersion": dispersion.tolist(), "method": methods.tolist(), "display_offset_points": OFFSETS[model_index]})
        axis.set_xlim(12, 180)
        axis.set_ylim(0, 1)
        axis.set_xticks(HOURS)
        axis.set_title(f"({chr(97 + index)}) {LABELS[parameter]}", loc="left", fontsize=9)
        ylabel = "Total Interest"
        if metric == "fss":
            ylabel = "FSS"
        axis.set_ylabel(ylabel)
        if index >= len(parameters) - 2:
            axis.set_xlabel("Lead Time (hours)")
        axis.grid(axis="y", color="0.84", linewidth=0.45)
        axis.set_axisbelow(True)
    names = []
    for model in MODELS:
        names.append(MODEL_LABELS[model])
    figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=4, fontsize=8)
    figure.canvas.draw()
    for axis in axes.flat:
        for hour in HOURS:
            center = axis.transData.transform((hour, 0))[0]
            half_width = 9 * figure.dpi / 72
            left = axis.transData.inverted().transform((center - half_width, 0))[0]
            right = axis.transData.inverted().transform((center + half_width, 0))[0]
            axis.axvspan(left, right, facecolor="0.94", edgecolor="none", zorder=0)
    RECORDS[name] = records
    save_figure(figure, name)


def plot_frequency_bias(parameters, name, all_rows):
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.94, hspace=0.52, wspace=0.30)
    upper = 3.5
    for row in all_rows:
        upper = max(upper, (row["mean"] + row["dispersion"]) * 1.15)
    records = []
    for model_index, model in enumerate(MODELS):
        means = []
        scales = []
        colors = []
        ticks = []
        for parameter in parameters:
            selected = []
            for row in all_rows:
                if row["model"] == model and row["parameter"] == parameter:
                    selected.append(row)
            assert len(selected) == 1
            row = selected[0]
            means.append(row["mean"])
            scales.append(row["dispersion"])
            colors.append(PARAMETER_COLORS[parameter])
            ticks.append(SHORT_LABELS[parameter])
            records.append(row)
        axis = axes.flat[model_index]
        x = np.arange(len(parameters))
        axis.bar(x, means, color=colors, edgecolor="black", linewidth=0.35, alpha=0.85)
        lower = np.minimum(scales, means)
        axis.errorbar(x, means, yerr=np.vstack([lower, scales]), fmt="none", ecolor="black", capsize=2, linewidth=0.7)
        axis.axhline(1, color="0.45", linestyle="--", linewidth=0.7)
        for index, mean in enumerate(means):
            label_y = mean + scales[index] + upper * 0.025
            if abs(label_y - 1) < upper * 0.04:
                label_y = 1 + upper * 0.04
            axis.text(index, label_y, f"{mean:.2f}", ha="center", fontsize=6.5)
        axis.set_ylim(0, upper)
        axis.set_xticks(x, ticks, rotation=60, ha="right", rotation_mode="anchor")
        axis.set_ylabel("Frequency Bias (FB)")
        axis.set_title(f"({chr(97 + model_index)}) {MODEL_LABELS[model]}, 168 h", loc="left", fontsize=9)
    RECORDS[name] = records
    save_figure(figure, name)


def plot_case_frequency_bias(parameters, name, data):
    models = MODELS + ["gfs"]
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.80)
    x = np.arange(len(parameters))
    width = 0.15
    offsets = np.linspace(-0.30, 0.30, 5)
    handles = []
    value_labels = []
    records = []
    for model_index, model in enumerate(models):
        values = []
        for parameter in parameters:
            value = data[model][parameter]
            assert np.isfinite(value)
            values.append(value)
            records.append({"model": model, "parameter": parameter, "fb": value})
        bars = axis.bar(x + offsets[model_index], values, width * 0.85, color=COLORS[model], edgecolor="white", linewidth=0.5, alpha=0.88, label=MODEL_LABELS[model])
        handles.append(bars)
        for bar, value in zip(bars, values):
            label = f"{value:.2f}"
            if abs(value) < 0.005:
                label = "0"
            color = "black"
            if value > 1.2:
                color = "red"
            elif value < 0.5:
                color = "blue"
            value_label = axis.text(bar.get_x() + bar.get_width() / 2, value + 0.07, label, rotation=90, ha="center", va="bottom", fontsize=7, color=color)
            value_labels.append(value_label)
    axis.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axis.axhspan(0, 0.5, alpha=0.07, facecolor="blue", edgecolor="none")
    axis.axhspan(1.5, 4.6, alpha=0.07, facecolor="red", edgecolor="none")
    ticks = []
    names = []
    for parameter in parameters:
        ticks.append(SHORT_LABELS[parameter])
    for model in models:
        names.append(MODEL_LABELS[model])
    axis.set_xticks(x, ticks, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_ylim(-0.08, 4.6)
    axis.set_ylabel("Frequency Bias (FB)")
    axis.grid(False)
    figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=5, fontsize=7.5)
    figure.text(0.53, 0.875, "19 May 2025 | 168 h Frequency Bias\nML models: ERA5 reference | GFS: GFS reference", ha="center", fontsize=9)
    shade_handles = [Patch(facecolor="blue", alpha=0.15, label="FB < 0.5"), Patch(facecolor="red", alpha=0.15, label="FB > 1.5")]
    figure.legend(handles=shade_handles, loc="lower center", bbox_to_anchor=(0.53, 0.01), ncol=2, fontsize=7)
    figure.canvas.draw()
    reference_y = axis.transData.transform((0, 1))[1]
    renderer = figure.canvas.get_renderer()
    for value_label in value_labels:
        bounds = value_label.get_window_extent(renderer)
        if bounds.y0 < reference_y < bounds.y1:
            x_position, y_position = value_label.get_position()
            value_label.set_position((x_position, 1.07))
    RECORDS[name] = records
    save_figure(figure, name)


def plot_case_mode(parameters, name, case_data):
    rows = len(parameters) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(7.2, rows * 2.6))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.075, top=0.91, hspace=0.48, wspace=0.27)
    handles = []
    records = []
    for index, parameter in enumerate(parameters):
        axis = axes.flat[index]
        for model in MODELS + ["gfs"]:
            path = ROOT / "2025-05-19-23UTC" / model / "MODE" / f"MODE_summary_{parameter}_{model}.csv"
            record_source(path)
            data = pd.read_csv(path).sort_values("Lead Time (h)")
            np.testing.assert_array_equal(data["Lead Time (h)"], HOURS)
            values = data["Mean Total Interest"].to_numpy()
            np.testing.assert_allclose(values, case_data[f"{model}/{parameter}"]["Mean Total Interest"], rtol=0, atol=0)
            mask = np.isfinite(values)
            line, = axis.plot(HOURS[mask], values[mask], color=COLORS[model], linewidth=1, marker="o", markersize=2.8, markerfacecolor=COLORS[model], markeredgecolor="white", markeredgewidth=0.4, label=MODEL_LABELS[model])
            if index == 0:
                handles.append(line)
            records.append({"model": model, "parameter": parameter, "hours": HOURS.tolist(), "total_interest": values.tolist()})
        axis.set_title(f"({chr(97 + index)}) {LABELS[parameter]}", loc="left", fontsize=9)
        axis.set_xticks(HOURS)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Total Interest")
        axis.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        if index >= len(parameters) - 2:
            axis.set_xlabel("Lead Time (hours)")
    names = []
    for model in MODELS + ["gfs"]:
        names.append(MODEL_LABELS[model])
    figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=5, fontsize=7.5)
    RECORDS[name] = records
    save_figure(figure, name)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    all_data = read_json(ROOT / "fig_revisions/output/unified_dispersion_20261016/source_data.json")
    case_records = read_json(ROOT / "fig_revisions/output/model_colors_20261019/mode_fb_verification.json")
    plot_performance(MAIN_PARAMETERS, "fig4_LFC_focus")
    plot_performance(ALL_PARAMETERS[:4], "figC1_performance_all_part1")
    plot_performance(ALL_PARAMETERS[4:], "figC2_performance_all_part2", first_letter=4)
    for metric, number in [("fss", 5), ("mode", 6)]:
        plot_score_grid(metric, MAIN_PARAMETERS, f"fig{number}_{metric}_LFC_focus")
        plot_score_grid(metric, ALL_PARAMETERS, f"figC{number - 2}_{metric}_all")
    plot_frequency_bias(MAIN_PARAMETERS, "fig7_fb_LFC_focus", all_data["frequency_bias"])
    plot_frequency_bias(ALL_PARAMETERS, "figC5_fb_all", all_data["frequency_bias"])
    plot_case_frequency_bias(MAIN_PARAMETERS, "fig12_case_fb_LFC_focus", case_records["frequency_bias_case"]["data"])
    plot_case_frequency_bias(ALL_PARAMETERS, "figC6_case_fb_all", case_records["frequency_bias_case"]["data"])
    plot_case_mode(MAIN_PARAMETERS, "fig13_case_mode_LFC_focus", case_records["mode_case"]["data"])
    plot_case_mode(ALL_PARAMETERS, "figC7_case_mode_all", case_records["mode_case"]["data"])
    (OUTPUT / "plotted_data.json").write_text(json.dumps(RECORDS, indent=2) + "\n")
    record_source(Path(performance.__file__))
    record_source(Path(performance.__file__).with_name("performance_diagram_label_layout.json"))
    record_source(Path(__file__).resolve())
    (OUTPUT / "source_sha256.json").write_text(json.dumps(SOURCES, indent=2) + "\n")


if __name__ == "__main__":
    main()
