"""Render the six retained Figure 4 panels with the existing reviewed callouts."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
import numpy as np


def _draw_performance_diagram(
        ax,
        all_models_data,
        parameter,
        show_legend,
        draw_ellipse=False,
        sigma=1,
        show_title=True):
    """
    Plot multiple models on one performance diagram (optional error ellipses)

    Parameters:
    -----------
    all_models_data : dict
        Dictionary containing data for each model, format:
        {
            model_name: {
                'lead_times': list,
                'mean_scores': array,
                'std_scores': array,
            }
        }
    parameter : str
        Parameter name (for title)
    show_legend : bool
        Whether to show legend
    draw_ellipse : bool
        Whether to draw error ellipses
    sigma : int | float
        Standard deviation multiplier for ellipses
    """
    # CSI contours
    x_grid = np.linspace(0.01, 1, 100)
    y_grid = np.linspace(0.01, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x_grid, y_grid)
    CSI_grid = 1 / (1 / SR_grid + 1 / POD_grid - 1)
    ax.contourf(SR_grid, POD_grid, CSI_grid, levels=np.arange(0, 1.1, 0.1),
                cmap='Greys', alpha=0.15)
    cs = ax.contour(
        SR_grid,
        POD_grid,
        CSI_grid,
        levels=np.arange(
            0.1,
            1.0,
            0.1),
        colors='silver',
        linewidths=0.8,
        linestyles='--')
    ax.clabel(cs, inline=True, fontsize=12, fmt='%.1f')

    # Bias lines
    bias_levels = [0.1, 0.3, 0.5, 0.8, 1, 1.25, 1.5, 2, 3, 5, 10]
    for b in bias_levels:
        if b == 1:
            ls = '-'
            alpha = 0.6
        else:
            ls = ':'
            alpha = 0.3
        if b > 1:
            x_bias = np.array([0, 1 / b])
        else:
            x_bias = np.array([0, 1])
        y_bias = b * x_bias
        ax.plot(x_bias, y_bias, color='black', linestyle=ls, alpha=alpha)

        # Keep the line at FB=10 but omit its crowded top-edge label.
        if b == 10:
            continue

        # Place labels at top or right edge of the plot
        if b <= 1:
            ax.text(
                1.01,
                b,
                f'FB={b}',
                verticalalignment='center',
                fontsize=12,
                alpha=0.6)
        else:
            label_x = 1 / b
            if b == 1.5:
                label_x -= 0.025
            if b == 1.25:
                label_x += 0.025
            ax.text(
                label_x,
                1.01,
                f'FB={b}',
                horizontalalignment='center',
                fontsize=12,
                alpha=0.6)

    # Define styles for different models
    model_styles = {
        'pangu': {
            'color': '#1f77b4',
            'marker': 'o',
            'line_style': '-',
            'label': 'Pangu',
            'zorder': 5},
        'fengwu_v1': {
            'color': '#2e8b57',
            'marker': 's',
            'line_style': '-',
            'label': 'FengWu v1',
            'zorder': 4},
        'fengwu_v2': {
            'color': '#C87500',
            'marker': '^',
            'line_style': '-',
            'label': 'FengWu v2',
            'zorder': 3},
        'fuxi': {
            'color': '#d62728',
            'marker': 'D',
            'line_style': '-',
            'label': 'FuXi',
            'zorder': 2},
    }

    # Store all trajectory line objects for legend
    lines = []
    labels = []

    # Iterate over each model to plot data
    for model_name, model_data in all_models_data.items():
        if model_name not in model_styles:
            continue

        lead_times = model_data['lead_times']
        mean_scores = model_data['mean_scores']
        std_scores = model_data.get('std_scores', None)

        # mean_scores/std_scores: (n_times,4)  column order [POD, CSI, FSS, SR]
        pod = mean_scores[:, 0]
        sr = mean_scores[:, 3]
        pod_std = None
        sr_std = None
        if std_scores is not None:
            pod_std = std_scores[:, 0]
            sr_std = std_scores[:, 3]

        n_points = len(lead_times)

        # Get model style
        color = model_styles[model_name]['color']
        marker = model_styles[model_name]['marker']
        line_style = model_styles[model_name]['line_style']
        label = model_styles[model_name]['label']
        zorder = model_styles[model_name]['zorder']

        # Plot trajectory line (connecting time points)
        line, = ax.plot(sr, pod, color=color, alpha=0.8, linewidth=2.5,
                        linestyle=line_style, marker='', zorder=zorder)

        # Plot individual time points (with white edge for visibility)
        ax.scatter(sr, pod, s=120, c=[color], marker=marker,
                   edgecolors='white', linewidths=1.5, zorder=zorder + 10)

        # Error ellipses
        if draw_ellipse and std_scores is not None:
            for i in range(n_points):
                if np.isnan(sr[i]) or np.isnan(pod[i]):
                    continue
                if np.isnan(sr_std[i]) or np.isnan(pod_std[i]):
                    continue
                if sr_std[i] <= 0 or pod_std[i] <= 0:
                    continue

                e = Ellipse(
                    (sr[i], pod[i]),
                    width=2 * sigma * sr_std[i],
                    height=2 * sigma * pod_std[i],
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.20,
                    zorder=zorder - 1
                )
                ax.add_patch(e)

        # Store a combined legend handle: colour + line style + marker shape
        legend_handle = Line2D(
            [], [], color=color, linestyle=line_style, marker=marker,
            markersize=10, markerfacecolor=color, markeredgecolor='white',
            linewidth=2.5, label=label)
        lines.append(legend_handle)
        labels.append(label)

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=16, fontweight='bold')
    ax.set_ylabel(
        "Probability of Detection (POD)",
        fontsize=16,
        fontweight='bold')
    if show_title:
        ax.set_title(parameter, fontsize=16, pad=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.3)

    # Add legend
    if show_legend:
        ax.legend(
            lines,
            labels,
            loc='upper right',
            fontsize=13,
            framealpha=0.9)


def _place_lead_time_labels(ax, all_models_data):
    """Draw the individually reviewed callouts in physical-point coordinates."""
    layout_path = Path(__file__).with_name('performance_diagram_label_layout.json')
    layouts = json.loads(layout_path.read_text())
    scale = ax.figure.dpi / 72
    colors = {'pangu': '#1f77b4', 'fengwu_v1': '#2e8b57', 'fengwu_v2': '#C87500', 'fuxi': '#d62728'}
    annotations = []
    layout_title = ax.get_title()
    if layout_title == '(e) LCPL':
        layout_title = '(f) LCPL'
    if layout_title == '(f) STP':
        layout_title = '(h) STP'
    for entry in layouts[layout_title]:
        data = all_models_data[entry['model']]
        index = data['lead_times'].index(entry['lead'])
        scores = data['mean_scores'][index]
        point = (float(scores[3]), float(scores[0]))
        color = colors[entry['model']]
        annotation = ax.annotate(entry['lead'], xy=point, xytext=entry['offset_pt'],
                                 textcoords='offset points', ha='center', va='center',
                                 fontsize=12, fontweight='bold', color=color, alpha=0.95,
                                 zorder=25, bbox=dict(boxstyle='round,pad=0.02', facecolor='none', edgecolor='none'))
        annotation.set_gid(f"lead-{entry['model']}-{entry['lead']}")
        annotation.set_in_layout(False)
        screen_point = ax.transData.transform(point)
        screen_vertices = screen_point + np.array(entry['leader_relative_pt']) * scale
        coordinates = ax.transData.inverted().transform(screen_vertices)
        line, = ax.plot(coordinates[:, 0], coordinates[:, 1], color=color,
                        linewidth=0.8, alpha=0.7, zorder=8, solid_capstyle='round')
        line.set_gid(f"leader-{entry['model']}-{entry['lead']}")
        annotations.append(annotation)
    return annotations



def main():
    root = Path(__file__).resolve().parents[2]
    output_dir = root / 'fig_revisions' / 'output' / 'model_colors_20261019'
    output_dir.mkdir(parents=True, exist_ok=True)
    models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
    lead_times = ['24h', '48h', '72h', '120h', '168h']
    parameters = [
        ('cape', '(a) CAPE'),
        ('cin', '(b) CIN'),
        ('k_idx', '(c) K Index'),
        ('positive_helicity', '(d) 0-3 km Positive SRH'),
        ('lcl', '(e) LCPL'),
        ('stp', '(f) STP')]
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'pdf.fonttype': 42,
        'svg.fonttype': 'none'})
    fig, axes_grid = plt.subplots(3, 2, figsize=(14.67, 18.30), dpi=300)
    axes = axes_grid.flatten()
    panels_data = []
    source_hashes = {}
    for panel_index, (parameter, title) in enumerate(parameters):
        all_models_data = {}
        for model in models:
            data_path = root / 'avg_scores' / model / f'scores_meanvar_{parameter}_{model}.npz'
            source_hashes[str(data_path.relative_to(root))] = hashlib.sha256(data_path.read_bytes()).hexdigest()
            with np.load(data_path) as data:
                mean_scores = data['mean'].copy()
                dispersion = data['dispersion'].copy()
            if mean_scores.shape != (5, 4):
                raise ValueError(f'Unexpected score shape: {data_path}: {mean_scores.shape}')
            if not np.isfinite(mean_scores).all():
                raise ValueError(f'Nonfinite scores: {data_path}')
            all_models_data[model] = {
                'lead_times': lead_times,
                'mean_scores': mean_scores,
                'std_scores': dispersion}
        ax = axes[panel_index]
        _draw_performance_diagram(ax, all_models_data, title, show_legend=True, show_title=False)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=28)
        panels_data.append(all_models_data)

    fig.tight_layout(h_pad=4, w_pad=5)
    fig.canvas.draw()
    all_annotations = []
    for ax, all_models_data in zip(axes, panels_data):
        annotations = _place_lead_time_labels(ax, all_models_data)
        all_annotations.extend(annotations)
    fig.canvas.draw()
    if len(all_annotations) != 120:
        raise AssertionError('Expected six panels with twenty lead-time labels each')

    # The rendered alignment check runs before any preview is exported.
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, json_out=output_dir / 'fig4_leadtime_preview.alignment.json', strict=True)
    output_base = output_dir / 'fig4_performance_consistent_colors'
    fig.savefig(output_base.with_suffix('.png'), dpi=300)
    fig.savefig(output_base.with_suffix('.pdf'))
    fig.savefig(output_base.with_suffix('.svg'))
    for panel_index, ax in enumerate(axes):
        panel_boxes = [ax.get_tightbbox(fig.canvas.get_renderer())]
        for annotation in ax.texts:
            if annotation.get_gid() and annotation.get_gid().startswith('lead-'):
                panel_boxes.append(annotation.get_window_extent(fig.canvas.get_renderer()))
        extent = Bbox.union(panel_boxes).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output_dir / f'panel_{parameters[panel_index][0]}.png', dpi=300,
                    bbox_inches=extent.padded(0.12))
    (output_dir / 'source_data_sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    plt.close(fig)
    print(f'Saved {output_base.with_suffix(".png")} (120 labels, 24 unchanged data files)')


if __name__ == '__main__':
    main()
