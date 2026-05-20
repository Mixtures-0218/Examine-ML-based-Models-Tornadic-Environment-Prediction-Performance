import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==================== Configuration ====================
# Specify the time to plot (modify this variable to plot different times)
time_to_plot = '2025-05-19-23UTC'

parameters = [
    'cape',
    'cin',
    'k_idx',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'lfc',
    'ehi']

models = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi', 'gfs']
lead_times = [24, 48, 72, 120, 168]

parameter_display = {
    'cape': 'CAPE',
    'cin': 'CIN',
    'k_idx': 'K Index',
    'positive_helicity': '0-3 km Positive SRH',
    'negative_helicity': '0-3 km Negative SRH',
    'lcl': 'LCL',
    'lfc': 'LFC',
    'ehi': 'EHI'
}

# Color scheme (add GFS)
model_colors = {
    'pangu': '#1f77b4',
    'fengwu_v1': '#2e8b57',
    'fengwu_v2': '#3cb371',
    'fuxi': '#d62728',
    'gfs': '#ff8c00',
}

# Model display names (add GFS)
model_labels = {
    'pangu': 'Pangu',
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'fuxi': 'FuXi',
    'gfs': 'GFS',
}

# Output directory
output_base = os.path.join('single_mode_plots', time_to_plot)
os.makedirs(output_base, exist_ok=True)


# ==================== Plotting Functions ====================
def _read_model_csv(time, model, param):
    """Read the MODE CSV file for a single time and model"""
    csv_path = os.path.join(
        time, model, 'MODE', f'MODE_summary_{param}_{model}.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df.sort_values('Lead Time (h)')


def plot_parameter(param, time):
    """Generate a figure with four subplots for a single parameter (single time, no error bars)"""
    display_name = parameter_display.get(param, param)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    param_idx = parameters.index(param) if param in parameters else 0
    param_letter = chr(ord('a') + param_idx)

    # Four subplot titles and corresponding data columns (use Mean columns for single time)
    subplots = [
        {
            'ax': axes[0],
            'title': 'Total Interest',
            'col': 'Mean Total Interest',
            'ylim': (0, 1)
        },
        {
            'ax': axes[1],
            'title': 'Mean Centroid Distance Deviation (grid cells)',
            'col': 'Mean Centroid Dist (grid)',
            'ylim': None
        },
        {
            'ax': axes[2],
            'title': 'Mean Area Ratio',
            'col': 'Mean Area Ratio',
            'ylim': (0, 1)
        },
        {
            'ax': axes[3],
            'title': 'Mean Angle Interest',
            'col': 'Mean Angle Interest',
            'ylim': (0, 1)
        }
    ]

    # Centroid Distance fixed y-axis range 0-40
    subplots[1]['ylim'] = (0, 40)

    # Plot curves for each model (no error bars, lines only)
    for model in models:
        df = _read_model_csv(time, model, param)
        if df is None:
            print(f"Warning: file not found, skipping {model}")
            continue

        x = df['Lead Time (h)'].values
        color = model_colors.get(model, '#333333')
        label = model_labels.get(model, model)

        for sp_idx, sp in enumerate(subplots):
            ax = sp['ax']
            col = sp['col']

            if col not in df.columns:
                continue

            y = df[col].values
            mask = ~np.isnan(y)
            if not any(mask):
                continue

            ax.plot(
                x[mask], y[mask],
                color=color,
                linewidth=2,
                marker='o',
                markersize=5,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=label if sp_idx == 0 else "")

    # Set subplot properties
    for sp_idx, sp in enumerate(subplots):
        ax = sp['ax']
        ax.set_xlabel('Lead Time (h)', fontsize=11, fontweight='bold')
        ax.set_ylabel(sp['title'], fontsize=11, fontweight='bold')
        ax.set_title(
            f"({param_letter}.{sp_idx + 1}) {sp['title']}",
            fontsize=12,
            fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_xticks(lead_times)
        ax.set_xticklabels([f'{h}h' for h in lead_times], fontsize=10)
        if sp['ylim'] is not None:
            ax.set_ylim(sp['ylim'])

    # Highlight Total Interest subplot background
    axes[0].set_facecolor('#f0f0f0')

    # Get legend from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(
        unique.values(),
        unique.keys(),
        loc='best',
        fontsize=9,
        framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_base, f'mode_plots_{param}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_total_interest(param, time):
    """Generate a Total Interest Index line plot (single time, no error bars)"""
    display_name = parameter_display.get(param, param)
    plt.figure(figsize=(12, 7), dpi=100)

    for model in models:
        df = _read_model_csv(time, model, param)
        if df is None:
            print(f"Warning: file not found, skipping {model}")
            continue

        if 'Mean Total Interest' not in df.columns:
            continue

        x = df['Lead Time (h)'].values
        y = df['Mean Total Interest'].values
        mask = ~np.isnan(y)
        if not any(mask):
            continue

        color = model_colors.get(model, '#333333')
        label = model_labels.get(model, model)

        plt.plot(
            x[mask], y[mask],
            color=color,
            linewidth=2,
            marker='o',
            markersize=5,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=0.8,
            label=label)

    plt.xlabel('Lead Time (h)', fontsize=11, fontweight='bold')
    plt.ylabel('Total Interest Index', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    plt.xticks(lead_times, [f'{h}h' for h in lead_times], fontsize=10)
    plt.ylim(0, 1)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(
        unique.values(),
        unique.keys(),
        loc='best',
        fontsize=9,
        framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_base, f'total_interest_{param}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _combine_images(image_paths, nrows, ncols, figsize, save_path,
                    dpi=200, titles=None, suptitle=None,
                    suptitle_y=0.98, tight_rect_top=0.95):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=suptitle_y)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(image_paths):
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=13, fontweight='bold', pad=10)
        else:
            ax.axis('off')
    if suptitle:
        plt.tight_layout(rect=[0, 0, 1, tight_rect_top])
    else:
        plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _layout_for_grid(nrows, ncols):
    if nrows >= 4:
        suptitle_y = 0.985
        gap = 0.020
    else:
        suptitle_y = 0.990
        gap = 0.018
    return suptitle_y, suptitle_y - gap


# ==================== Main Loop ====================
print(f"========== Plotting time: {time_to_plot} ==========")
print(f"Output directory: {output_base}")

# 1. Generate individual subplots for each parameter
for param in parameters:
    plot_parameter(param, time_to_plot)
    plot_total_interest(param, time_to_plot)

# 2. Combine eight parameter figures into one (4x2 and 2x4)
param_paths = [
    (p, os.path.join(output_base, f'mode_plots_{p}.png'))
    for p in parameters
]
param_paths = [(p, path) for p, path in param_paths if os.path.exists(path)]

if param_paths:
    combined_paths = [path for _, path in param_paths]
    titles = [
        f"({chr(ord('a') + i)}) {parameter_display.get(p, p)}"
        for i, (p, _) in enumerate(param_paths)
    ]
    suptitle = (
        f"MODE of ML-based Models and GFS for "
        f"Tornado Relevant Parameters\n{time_to_plot}")

    combined_save_4x2 = os.path.join(output_base, 'mode_plots_all_4x2.png')
    y_4x2, top_4x2 = _layout_for_grid(4, 2)
    _combine_images(
        combined_paths, 4, 2, (20, 28),
        combined_save_4x2, dpi=200, titles=titles, suptitle=suptitle,
        suptitle_y=y_4x2, tight_rect_top=top_4x2)
    print(f"Saved combined figure: {combined_save_4x2}")

    combined_save_2x4 = os.path.join(output_base, 'mode_plots_all_2x4.png')
    y_2x4, top_2x4 = _layout_for_grid(2, 4)
    _combine_images(
        combined_paths, 2, 4, (28, 16),
        combined_save_2x4, dpi=200, titles=titles, suptitle=suptitle,
        suptitle_y=y_2x4, tight_rect_top=top_2x4)
    print(f"Saved combined figure: {combined_save_2x4}")

# 3. Combine Total Interest Index (4x2 and 2x4)
interest_paths = [
    (p, os.path.join(output_base, f'total_interest_{p}.png'))
    for p in parameters
]
interest_paths = [
    (p, path) for p, path in interest_paths if os.path.exists(path)
]

if interest_paths:
    interest_img_paths = [path for _, path in interest_paths]
    interest_titles = [
        f"({chr(ord('a') + i)}) {parameter_display.get(p, p)}"
        for i, (p, _) in enumerate(interest_paths)
    ]
    interest_suptitle = (
        f"Total Interest Index of ML-based Models and GFS for "
        f"Tornado Relevant Parameters\n{time_to_plot}")

    interest_save_4x2 = os.path.join(
        output_base, 'total_interest_all_4x2.png')
    y_4x2, top_4x2 = _layout_for_grid(4, 2)
    _combine_images(
        interest_img_paths, 4, 2, (20, 28),
        interest_save_4x2, dpi=200, titles=interest_titles,
        suptitle=interest_suptitle,
        suptitle_y=y_4x2, tight_rect_top=top_4x2)
    print(f"Saved combined figure: {interest_save_4x2}")

    interest_save_2x4 = os.path.join(
        output_base, 'total_interest_all_2x4.png')
    y_2x4, top_2x4 = _layout_for_grid(2, 4)
    _combine_images(
        interest_img_paths, 2, 4, (28, 16),
        interest_save_2x4, dpi=200, titles=interest_titles,
        suptitle=interest_suptitle,
        suptitle_y=y_2x4, tight_rect_top=top_2x4)
    print(f"Saved combined figure: {interest_save_2x4}")

print(f"All parameter plots completed! Time: {time_to_plot}")
