"""Preview layout revisions for manuscript Figures 11 and 13.

Figure 11 deliberately preserves the current manuscript's reconstructed theta-e
coordinates. This script does not validate or replace those scientific values.
Figure 15 is recovered from the existing single-case plot, without redrawing it.
Run with the nature-figure scripts directory on PYTHONPATH for alignment QA.
"""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from cartopy import crs as ccrs, feature as cfeature
from matplotlib.text import Text
from scipy.stats import linregress, pearsonr

from audit_panel_alignment import require_matplotlib_panel_alignment


BASE = Path('/Volumes/T7/Research/spatial_evaluation')
OUTPUT = BASE / 'fig_revisions/output/selected_figures_preview'
CASE = '2025-05-19-23UTC'
SOURCE_SCRIPT = BASE / 'fig_revisions/scripts/rebuild_fig12_fig14.py'
SOURCE_HASHES = {}


def record_source(path):
    SOURCE_HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def load_array(path):
    record_source(path)
    return np.load(path)


def export_figure(fig, name, plot_axes):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=plot_axes, json_out=OUTPUT / (name + '.alignment.json')
    )
    fig.savefig(OUTPUT / (name + '.png'), dpi=300)
    fig.savefig(OUTPUT / (name + '.pdf'))
    fig.savefig(OUTPUT / (name + '.svg'))


def figure11():
    """Change date-label placement while preserving existing plotted values."""
    record_source(SOURCE_SCRIPT)
    specification = importlib.util.spec_from_file_location('legacy', SOURCE_SCRIPT)
    legacy = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(legacy)

    coordinates = []
    cape_biases = []
    lfc_biases = []
    labels = []
    for case in legacy.CASES:
        coordinates.append(legacy.ORIG_DTHETA[case])
        for parameter in ['cape', 'lfc']:
            reference = load_array(BASE / case / (parameter + '_real.npy'))
            forecast = load_array(
                BASE / case / 'fuxi' / (parameter + '_fuxi.npy')
            )[0]
            bias = float(np.nanmean(forecast - reference))
            if parameter == 'cape':
                cape_biases.append(bias)
            else:
                lfc_biases.append(bias)
        year, month, day, _ = case.split('-')
        month_names = {'04': 'Apr', '05': 'May'}
        labels.append(year + '\n' + month_names[month] + ' ' + day)

    x_values = np.array(coordinates)
    offsets = [
        [(12, 24), (-44, 15), (42, 10), (12, -28), (12, -26),
         (-48, 10), (12, -26), (30, -12), (14, -26), (18, 24)],
        [(12, 24), (36, -6), (-46, -8), (14, 24), (12, -25),
         (-50, -4), (-16, 24), (-16, 24), (18, -26), (26, 16)],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.85, wspace=0.28)
    panels = [
        ('CAPE', cape_biases, '#1f77b4', (-500, 1080), r'$\Delta$CAPE (J kg$^{-1}$)'),
        ('LFCP', lfc_biases, '#ff7f0e', (-45, 100), r'$\Delta$LFCP (hPa)'),
    ]
    audit = {'coordinate_provenance': 'legacy.ORIG_DTHETA; reconstructed from old figure',
             'cases': legacy.CASES, 'theta_e': coordinates, 'panels': []}
    for panel_index, (name, values, color, limits, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        y_values = np.array(values)
        correlation, probability = pearsonr(x_values, y_values)
        regression = linregress(x_values, y_values)
        ax.scatter(x_values, y_values, s=65, color=color, edgecolors='#333333',
                   linewidths=0.65, zorder=4)
        fit_x = np.array([x_values.min() - 0.5, x_values.max() + 0.5])
        ax.plot(fit_x, regression.intercept + regression.slope * fit_x,
                '--', color='#777777', linewidth=1.2, zorder=2)
        ax.axhline(0, color='#888888', linewidth=0.7)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(*limits)
        ax.set_xlabel(r'$\Delta\theta_e$ at 1000 hPa (K)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title('(' + chr(97 + panel_index) + ') ' + name, loc='left',
                     fontsize=14, fontweight='bold', pad=26)
        ax.text(0, 1.035,
                f'r = {correlation:.3f}    p = {probability:.4f}    '
                f'R$^2$ = {correlation ** 2:.2f}    n = 10',
                transform=ax.transAxes, fontsize=10, color='#444444')
        ax.grid(alpha=0.16, linestyle='--', zorder=0)
        ax.tick_params(labelsize=10)
        annotations = []
        for case_index, label in enumerate(labels):
            annotation = ax.annotate(
                label, (x_values[case_index], y_values[case_index]),
                xytext=offsets[panel_index][case_index], textcoords='offset points',
                fontsize=10, ha='center', va='center', color='#333333', zorder=5,
                bbox={'boxstyle': 'square,pad=0.18', 'fc': 'white', 'ec': 'none'},
                arrowprops={'arrowstyle': '-', 'color': '#777777', 'lw': 0.75,
                            'shrinkA': 3, 'shrinkB': 5},
            )
            annotation.set_gid('case-' + legacy.CASES[case_index])
            annotations.append(annotation)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for annotation in annotations:
            boxes.append(Text.get_window_extent(annotation, renderer).padded(2))
        for first in range(len(boxes)):
            assert ax.bbox.contains(boxes[first].x0, boxes[first].y0)
            assert ax.bbox.contains(boxes[first].x1, boxes[first].y1)
            for second in range(first + 1, len(boxes)):
                assert not boxes[first].overlaps(boxes[second]), (
                    name, labels[first], labels[second])
            marker_clearance = 6 * fig.dpi / 72
            for x_value, y_value in zip(x_values, y_values):
                pixel_x, pixel_y = ax.transData.transform((x_value, y_value))
                assert not boxes[first].padded(marker_clearance).contains(pixel_x, pixel_y), (
                    name, labels[first], 'label covers a data marker')
        audit['panels'].append({'parameter': name, 'biases': values,
                                'r': float(correlation), 'p': float(probability),
                                'label_collisions': 0})
    fig.text(0.5, 0.965, 'FuXi 24 h minus ERA5: case-mean differences',
             ha='center', fontsize=14)
    export_figure(fig, 'fig11_cape_lfc_theta_e_preview', axes)
    (OUTPUT / 'fig11_data_and_labels.json').write_text(json.dumps(audit, indent=2))
    plt.close(fig)


def figure13():
    """Use identical STP levels for seven fields and a shared colorbar."""
    directory = BASE / CASE
    reference = load_array(directory / 'stp_real.npy')
    gfs = load_array(directory / 'gfs/stp_gfs.npy')
    panels = [('ERA5', reference, 0), ('GFS reference (+5 h)', gfs[0], 1),
              ('GFS forecast (+29 h)', gfs[1], 2)]
    models = [('Pangu', 'pangu'), ('FengWu v1', 'fengwu_v1'),
              ('FengWu v2', 'fengwu_v2'), ('FuXi', 'fuxi')]
    for index, (display, model) in enumerate(models):
        data = load_array(directory / model / ('stp_' + model + '.npy'))[0]
        panels.append((display + ' (+24 h)', data, index + 4))

    maximum = 0.0
    for _, data, _ in panels:
        assert data.shape == reference.shape
        assert np.isfinite(data).all()
        assert np.min(data) >= 0
        maximum = max(maximum, float(np.max(data)))
    levels = np.linspace(0, np.ceil(maximum), 25)
    threshold = float(np.nanpercentile(reference, 75))
    longitude, latitude = np.meshgrid(
        np.linspace(260, 266, reference.shape[1]),
        np.linspace(38, 33, reference.shape[0]),
    )
    fig = plt.figure(figsize=(13.4, 7.5))
    grid = fig.add_gridspec(2, 4, left=0.055, right=0.985, bottom=0.20,
                           top=0.84, wspace=0.22, hspace=0.29)
    projection = ccrs.PlateCarree()
    audit_panels = []
    map_axes = []
    for index, (title, data, slot) in enumerate(panels):
        ax = fig.add_subplot(grid[slot // 4, slot % 4], projection=projection)
        map_axes.append(ax)
        ax.set_extent([-100, -94, 33, 38], crs=projection)
        filled = ax.contourf(longitude, latitude, data, levels=levels,
                             cmap='coolwarm', transform=projection)
        ax.add_feature(cfeature.STATES, edgecolor='#444444', linewidth=0.55)
        lines = ax.gridlines(draw_labels=True, xlocs=[-100, -98, -96, -94],
                             ylocs=[33, 35, 37], color='#666666',
                             linewidth=0.4, linestyle=':', alpha=0.25)
        lines.top_labels = False
        lines.right_labels = False
        lines.xlabel_style = {'size': 9}
        lines.ylabel_style = {'size': 9}
        contour = ax.contour(longitude, latitude, data, levels=[threshold],
                             transform=projection, colors='red', linewidths=1.1)
        contour.set_path_effects([path_effects.Stroke(linewidth=2.1, foreground='white'),
                                  path_effects.Normal()])
        ax.set_title('(' + chr(97 + index) + ') ' + title, fontsize=11,
                     fontweight='bold', loc='left', pad=10)
        audit_panels.append({'title': title, 'minimum': float(np.min(data)),
                             'maximum': float(np.max(data)),
                             'levels': filled.levels.tolist(),
                             'colormap': filled.cmap.name,
                             'clim': list(filled.get_clim())})
    note = fig.add_subplot(grid[0, 3])
    note.set_in_layout(False)
    note.axis('off')
    note.text(0.03, 0.78, 'Shared STP scale', fontsize=13, fontweight='bold')
    note.text(0.03, 0.59, 'ML forecast lead time: 24 h\nGFS forecast lead time: 29 h',
               fontsize=11, linespacing=1.6)
    note.text(0.03, 0.29, f'Red contour: STP = {threshold:.2f}\n(ERA5 75th percentile)',
               fontsize=11, linespacing=1.6, color='#b2182b')
    colorbar_axis = fig.add_axes([0.21, 0.093, 0.58, 0.022])
    colorbar = fig.colorbar(filled, cax=colorbar_axis, orientation='horizontal',
                           ticks=np.arange(0, np.ceil(maximum) + 1, 2))
    colorbar.set_label('Significant Tornado Parameter (STP)', fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    fig.suptitle('STP spatial comparison | 19 May 2025, 23:00 UTC', fontsize=15, y=0.94)
    # The explanatory cell and shared colorbar are not comparable map panels.
    export_figure(fig, 'fig13_stp_lead24h_preview', map_axes)
    audit = {'shared_levels': levels.tolist(), 'red_contour_threshold': threshold,
             'panels': audit_panels}
    (OUTPUT / 'fig13_shared_scale.json').write_text(json.dumps(audit, indent=2))
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    figure11()
    figure13()
    recovered = BASE / 'single_mode_plots' / CASE / 'total_interest_all_4x2.png'
    record_source(recovered)
    shutil.copy2(recovered, OUTPUT / 'fig15_mode_case_recovered.png')
    (OUTPUT / 'source_data_sha256.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
    print('Saved Figure 11/13 previews and recovered Figure 15 to', OUTPUT)


if __name__ == '__main__':
    main()
