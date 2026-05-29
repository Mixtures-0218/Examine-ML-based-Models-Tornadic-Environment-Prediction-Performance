import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import os


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
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')

    # Bias lines
    bias_levels = [0.1, 0.3, 0.5, 0.8, 1, 1.25, 1.5, 2, 3, 5, 10]
    for b in bias_levels:
        ls = '-' if b == 1 else ':'
        alpha = 0.6 if b == 1 else 0.3
        x_bias = np.array([0, 1])
        y_bias = b * x_bias
        ax.plot(x_bias, y_bias, color='black', linestyle=ls, alpha=alpha)

        # Place labels at top or right edge of the plot
        if b <= 1:
            ax.text(
                1.01,
                b,
                f'FB={b}',
                verticalalignment='center',
                fontsize=8,
                alpha=0.6)
        else:
            ax.text(
                1 / b,
                1.01,
                f'FB={b}',
                horizontalalignment='center',
                fontsize=8,
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
            'color': '#3cb371',
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
        pod_std = std_scores[:, 0] if std_scores is not None else None
        sr_std = std_scores[:, 3] if std_scores is not None else None

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

        # Add time annotations
        for i in range(n_points):
            ax.annotate(
                lead_times[i],
                (sr[i],
                 pod[i]),
                xytext=(
                    5,
                    5),
                textcoords='offset points',
                fontsize=7,
                fontweight='bold',
                color=color,
                alpha=0.7,
                zorder=zorder + 5)

        # Store for legend
        lines.append(line)
        labels.append(label)

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=12, fontweight='bold')
    ax.set_ylabel(
        "Probability of Detection (POD)",
        fontsize=12,
        fontweight='bold')
    if show_title:
        ax.set_title(parameter, fontsize=12, pad=12)
    ax.grid(True, linestyle=':', alpha=0.3)

    # Add legend
    if show_legend:
        ax.legend(
            lines,
            labels,
            loc='upper right',
            fontsize=10,
            framealpha=0.9)


def performance_diagram_multiple_models(
        all_models_data,
        parameter,
        save_path=None,
        show=False,
        draw_ellipse=False,
        sigma=1):
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
    save_path : str | None
        Save path (None = do not save)
    show : bool
        Whether to display the figure
    draw_ellipse : bool
        Whether to draw error ellipses
    sigma : int | float
        Standard deviation multiplier for ellipses
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_performance_diagram(ax, all_models_data, parameter, show_legend=True,
                              draw_ellipse=draw_ellipse, sigma=sigma, show_title=False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def _combine_images(image_paths, nrows, ncols, figsize, save_path, title=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    parameters = [
        'cape',
        'cin',
        'k_idx',
        'positive_helicity',
        'negative_helicity',
        'lcl',
        'lfc',
        'stp']
    parameter_display = {
        'cape': 'CAPE',
        'cin': 'CIN',
        'k_idx': 'K Index',
        'positive_helicity': '0-3 km Positive SRH',
        'negative_helicity': '0-3 km Negative SRH',
        'lcl': 'LCL',
        'lfc': 'LFC',
        'stp': 'STP'
    }
    for i, ax in enumerate(axes):
        if i < len(image_paths):
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.set_title(f'{letters[i]} {parameter_display[parameters[i]]}', fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _resolve_data_path(model, parameter, primary_root, fallback_root):
    filename = f'scores_meanvar_{parameter}_{model}.npz'
    primary = os.path.join(primary_root, model, filename)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(fallback_root, model, filename)
    if os.path.exists(fallback):
        if primary_root != fallback_root:
            print(f"  Warning: {model} using fallback data directory: {fallback_root}")
        return fallback
    return None


# Define models and parameters to compare
models_ai = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
parameters = [
    'cape',
    'cin',
    'k_idx',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'lfc',
    'stp']
lead_times = ['24h', '48h', '72h', '120h', '168h']

parameter_display = {
    'cape': 'CAPE',
    'cin': 'CIN',
    'k_idx': 'K Index',
    'positive_helicity': '0-3 km Positive SRH',
    'negative_helicity': '0-3 km Negative SRH',
    'lcl': 'LCL',
    'lfc': 'LFC',
    'stp': 'STP'
}

output_base = 'avg_performance_diagram'
os.makedirs(output_base, exist_ok=True)

# Plot groups: AI only (10 times)
# ML+GFS (8 time points) requires ML means computed by removing the two 2013 time points

plot_groups = [
    {
        'name': 'ai_10',
        'models': models_ai,
        'title_suffix': 'AI (10 times)',
        'data_root': 'avg_scores'
    }
]


# Global switch for error ellipses
DRAW_ELLIPSE = False
ELLIPSE_SIGMA = 1

for group in plot_groups:
    group_name = group['name']
    group_models = group['models']
    group_title = group['title_suffix']
    data_root = group['data_root']

    output_dir = os.path.join(output_base, group_name)
    os.makedirs(output_dir, exist_ok=True)

    for parameter in parameters:
        # Load data for all models
        all_models_data = {}

        print(f"Loading model data: {parameter} | {group_title}")
        print("-" * 50)

        for model in group_models:
            try:
                data_path = _resolve_data_path(
                    model, parameter, data_root, data_root)

                if not data_path:
                    raise FileNotFoundError

                data = np.load(data_path)

                mean_scores = data['mean']
                std_scores = data['dispersion']

                print(f"o {model}: successfully loaded {len(lead_times)} time points")
                print(f"  mean_scores shape: {mean_scores.shape}")

                all_models_data[model] = {
                    'lead_times': lead_times,
                    'mean_scores': mean_scores,
                    'std_scores': std_scores
                }

            except FileNotFoundError:
                print(f"x {model}: data file not found")
                print(f"  Expected path: {data_path}")
            except Exception as e:
                print(f"x {model}: error loading data - {e}")

        print("-" * 50)

        # Check if any data was loaded successfully
        if all_models_data:
            print(f"Successfully loaded data for {len(all_models_data)} models")
            title = f"Mean Performance Diagrams of ML-based Models for Tornado-Relavent Parameters"
            save_path = os.path.join(
                output_dir, f'performance_diagram_{parameter}.png')
            performance_diagram_multiple_models(
                all_models_data,
                title,
                save_path=save_path,
                show=False,
                draw_ellipse=DRAW_ELLIPSE,
                sigma=ELLIPSE_SIGMA)
            print(f"Saved: {save_path}")
        else:
            print("No model data loaded successfully, please check file paths")

    # Combine eight figures into one (4x2 and 2x4)
    combined_paths = [
        os.path.join(
            output_dir,
            f'performance_diagram_{p}.png') for p in parameters]
    combined_paths = [p for p in combined_paths if os.path.exists(p)]
    if combined_paths:
        overall_title = "Mean Performance Diagrams of ML-based Models for Tornado-Relevant Parameters"

        combined_save_4x2 = os.path.join(
            output_dir, 'performance_diagram_all_4x2.png')
        _combine_images(combined_paths, 4, 2, (18, 26), combined_save_4x2, title=overall_title)
        print(f"Saved combined figure: {combined_save_4x2}")

        combined_save_2x4 = os.path.join(
            output_dir, 'performance_diagram_all_2x4.png')
        _combine_images(combined_paths, 2, 4, (26, 14), combined_save_2x4, title=overall_title)
        print(f"Saved combined figure: {combined_save_2x4}")
