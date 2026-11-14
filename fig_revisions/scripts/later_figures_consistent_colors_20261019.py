"""Change model colors in existing later figures, preserving original geometry."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import fitz

from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "fig_revisions/scripts"
OUTPUT = ROOT / "fig_revisions/output/model_colors_20261019"
ORIGINAL = OUTPUT / "original_render_check"
SUBMISSION = Path("/Users/luoyiding/Library/Mobile Documents/com~apple~CloudDocs/UW/Research/Papers/Current Works/ML-Weather-Evaluation/arXiv_submission/figures")
CASE = "2025-05-19-23UTC"
COLOR_CHANGES = {"#8bcf8b": "#2e8b57", "#42949e": "#C87500", "#3cb371": "#C87500"}
RECORDS = {}


def load_definitions(path):
    """Exclude legacy top-level pipelines and directory-creation calls."""
    source_text = path.read_text()
    if path.name == "MODE_plot_single.py":
        # Restore the verified 200-DPI layout renderer used for the existing PNGs.
        original = "plt.figure(figsize=(12, 7), dpi=100)"
        assert source_text.count(original) == 1
        source_text = source_text.replace(original, "plt.figure(figsize=(12, 7), dpi=200)", 1)
    source = ast.parse(source_text)
    nodes = []
    for node in source.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)):
            nodes.append(node)
    namespace = {"__file__": str(path), "__name__": "color_revision_source"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    RECORDS[str(path.name)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return namespace


def capture_figure(function, *arguments):
    before = set(plt.get_fignums())
    original_close = plt.close

    def keep_open(*args, **kwargs):
        return None

    plt.close = keep_open
    try:
        function(*arguments)
    finally:
        plt.close = original_close
    created = set(plt.get_fignums()) - before
    assert len(created) == 1, created
    number = created.pop()
    return plt.figure(number)


def geometry_signature(figure):
    """Hash coordinates, text and axes limits, independently of model colors."""
    records = []
    for axes in figure.axes:
        records.append([list(axes.get_position().bounds), list(axes.get_xlim()), list(axes.get_ylim())])
        for line in axes.lines:
            records.append([np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist(), line.get_marker(), line.get_linestyle()])
        for patch in axes.patches:
            records.append(patch.get_path().vertices.tolist())
        for text in axes.texts:
            records.append([text.get_text(), list(text.get_position()), text.get_fontsize()])
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def remap_color(value):
    key = to_hex(value).lower()
    if key in COLOR_CHANGES:
        return COLOR_CHANGES[key]
    return value


def recolor(figure):
    before = geometry_signature(figure)
    counts = {"lines": 0, "patches": 0}
    for artist in figure.findobj():
        if isinstance(artist, Line2D):
            old = to_hex(artist.get_color()).lower()
            if old in COLOR_CHANGES:
                artist.set_color(COLOR_CHANGES[old])
                counts["lines"] += 1
            for getter, setter in [(artist.get_markerfacecolor, artist.set_markerfacecolor), (artist.get_markeredgecolor, artist.set_markeredgecolor)]:
                value = getter()
                if isinstance(value, str) and value in ["auto", "none", "None"]:
                    continue
                setter(remap_color(value))
        elif isinstance(artist, Patch):
            for getter, setter in [(artist.get_facecolor, artist.set_facecolor), (artist.get_edgecolor, artist.set_edgecolor)]:
                value = getter()
                key = to_hex(value).lower()
                if key in COLOR_CHANGES:
                    replacement = list(matplotlib.colors.to_rgb(COLOR_CHANGES[key]))
                    replacement.append(value[3])
                    setter(replacement)
                    counts["patches"] += 1
    assert geometry_signature(figure) == before
    return {"geometry_sha256_before_and_after": before, "recolored_artists": counts}


def verify_original_png(generated, existing, name):
    first = np.asarray(Image.open(generated).convert("RGBA"))
    second = np.asarray(Image.open(existing).convert("RGBA"))
    assert first.shape == second.shape, (name, first.shape, second.shape)
    assert np.array_equal(first, second), name
    RECORDS[name] = {"original_render_pixel_identical": True}


def export(figure, name, dpi, bbox_inches="tight"):
    figure.canvas.draw()
    require_matplotlib_panel_alignment(figure, json_out=OUTPUT / f"{name}.alignment.json", strict=True)
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=dpi, bbox_inches=bbox_inches)
    plt.close(figure)


def profiles():
    with plt.rc_context(matplotlib.rcParamsDefault):
        namespace = load_definitions(SCRIPTS / "rebuild_analysis_figs.py")
        namespace["OUT"] = str(ORIGINAL)
        figure = capture_figure(namespace["fig10_profiles"])
        verify_original_png(ORIGINAL / "fig_multi_model_profiles.png", SUBMISSION / "fig10_profiles.png", "profiles")
        RECORDS["profiles"].update(recolor(figure))
        export(figure, "fig10_profiles_consistent_colors", 200)


def spatial_sd_ratio():
    with plt.rc_context(matplotlib.rcParamsDefault):
        namespace = load_definitions(SCRIPTS / "paper_dispersion_figures.py")
        namespace["OUTPUT"] = ORIGINAL
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
        figure = capture_figure(namespace["plot_spatial_sd_ratio"])
        old = fitz.open(SUBMISSION / "fig12_cape_sd_ratio_appendix_b.pdf")
        generated = fitz.open(ORIGINAL / "fig12_cape_sd_ratio_appendix_b.pdf")
        assert old[0].get_pixmap().samples == generated[0].get_pixmap().samples
        RECORDS["spatial_sd_ratio"] = {"original_render_pixel_identical": True}
        RECORDS["spatial_sd_ratio"].update(recolor(figure))
        export(figure, "fig12_cape_sd_ratio_consistent_colors", 300, bbox_inches=None)
        RECORDS["spatial_sd_ratio"]["data"] = namespace["RECORDS"]["spatial_sd_ratio"]
        old.close()
        generated.close()


def frequency_bias_case():
    with plt.rc_context(matplotlib.rcParamsDefault):
        namespace = load_definitions(SCRIPTS / "fig_2025_case_fb.py")
        namespace["OUT_DIR"] = str(ORIGINAL)
        data = namespace["compute_168h_fb"]()
        figure = capture_figure(namespace["plot_2025_fb_bars"], data)
        verify_original_png(ORIGINAL / "fig_2025_case_168h_fb.png", SUBMISSION / "fig14_fb_168h_case.png", "frequency_bias_case")
        RECORDS["frequency_bias_case"].update(recolor(figure))
        RECORDS["frequency_bias_case"]["data"] = data
        export(figure, "fig14_fb_168h_case_consistent_colors", 200)


def mode_case():
    with plt.rc_context(matplotlib.rcParamsDefault):
        # Both layout stages used a 200-DPI renderer in the existing figure.
        plt.rcParams["figure.dpi"] = 200
        namespace = load_definitions(ROOT / "MODE_plot_single.py")
        namespace["output_base"] = str(ORIGINAL)
        figures = []
        image_paths = []
        titles = []
        data = {}
        for index, parameter in enumerate(namespace["parameters"]):
            for model in namespace["models"]:
                frame = namespace["_read_model_csv"](str(ROOT / CASE), model, parameter)
                assert frame is not None, (model, parameter)
                data[f"{model}/{parameter}"] = frame.to_dict(orient="list")
            figure = capture_figure(namespace["plot_total_interest"], parameter, str(ROOT / CASE))
            verify_original_png(ORIGINAL / f"total_interest_{parameter}.png", ROOT / "single_mode_plots" / CASE / f"total_interest_{parameter}.png", f"mode_panel_{parameter}")
            figures.append(figure)
            image_paths.append(str(ORIGINAL / f"total_interest_{parameter}.png"))
            titles.append(f"({chr(97 + index)}) {namespace['parameter_display'][parameter]}")
        title = f"Total Interest Index of ML-based Models and GFS for Tornado Relevant Parameters\n{CASE}"
        y, top = namespace["_layout_for_grid"](4, 2)
        original_path = ORIGINAL / "total_interest_all_4x2.png"
        namespace["_combine_images"](image_paths, 4, 2, (20, 28), str(original_path), dpi=200, titles=titles, suptitle=title, suptitle_y=y, tight_rect_top=top)
        verify_original_png(original_path, SUBMISSION / "fig15_mode_case_corrected.png", "mode_case")
        audits = []
        for figure, path in zip(figures, image_paths):
            audits.append(recolor(figure))
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
        target = OUTPUT / "fig15_mode_case_consistent_colors.png"
        figure = capture_figure(namespace["_combine_images"], image_paths, 4, 2, (20, 28), str(target), 200, titles, title, y, top)
        RECORDS["mode_case"]["panels"] = audits
        RECORDS["mode_case"]["data"] = data
        export(figure, "fig15_mode_case_consistent_colors", 200)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ORIGINAL.mkdir(exist_ok=True)
    spatial_sd_ratio()
    frequency_bias_case()
    mode_case()
    profiles()
    (OUTPUT / "later_figure_data_and_verification.json").write_text(json.dumps(RECORDS, indent=2) + "\n")
    print("Saved four later figures; all original renders and plot geometry verified.")


if __name__ == "__main__":
    main()
