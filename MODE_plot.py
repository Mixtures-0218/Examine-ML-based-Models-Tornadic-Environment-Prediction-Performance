import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==================== Configuration ====================
parameters = [
    'cape',
    'cin',
    'k_idx',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'lfc',
    'stp']
models_ai = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
lead_times = [24, 48, 72, 120, 168]

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

# Color scheme
model_colors = {
    'pangu': '#1f77b4',
    'fengwu_v1': '#2e8b57',
    'fengwu_v2': '#3cb371',
    'fuxi': '#d62728',
}

# Model display names
model_labels = {
    'pangu': 'Pangu',
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'fuxi': 'FuXi',
}

# Input and output directories
data_root_base = 'avg_mode'
output_base = os.path.join('avg_mode', 'plots')
os.makedirs(output_base, exist_ok=True)

# Plot groups: AI 10 times on same figure, AI+GFS 8 times on same figure
# AI+GFS (8 times) requires AI means computed after removing the two 2013 time points

plot_groups = [
    {
        'name': 'ai_10',
        'models': models_ai,
        'title_suffix': 'AI (10 times)',
        'data_root': data_root_base
    }
]



def _resolve_csv_path(model, param, primary_root, fallback_root):
    filename = f'mode_summary_{param}_{model}.csv'
    primary = os.path.join(primary_root, model, filename)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(fallback_root, model, filename)
    if os.path.exists(fallback):
        if primary_root != fallback_root:
            print(f"Warning: {model} using fallback data directory: {fallback_root}")
        return fallback
    return None


# ==================== Plotting Functions ====================
def plot_parameter(
        param,
        models,
        labels_map,
        colors_map,
        data_root,
        output_dir,
        title_suffix):
    """Generate a 3x2 figure for a single parameter: Total Interest, Centroid/Boundary Dist, Angle, Area Ratio, Int Area"""
    display_name = parameter_display.get(param, param)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    axes = axes.flatten()

    param_idx = parameters.index(param) if param in parameters else 0
    param_letter = chr(ord('a') + param_idx)

    # Six subplot titles and corresponding data columns
    subplots = [
        {
            'ax': axes[0],
            'title': 'Total Interest',
            'mean_col': 'interest_mean',
            'ylim': (0, 1)
        },
        {
            'ax': axes[1],
            'title': 'Centroid Distance (grid cells)',
            'mean_col': 'cent_dist_mean',
            'ylim': (0, 35)
        },
        {
            'ax': axes[2],
            'title': 'Boundary Distance (grid cells)',
            'mean_col': 'bound_dist_mean',
            'ylim': (0, 35)
        },
        {
            'ax': axes[3],
            'title': 'Angle Interest',
            'mean_col': 'angle_mean',
            'ylim': (0, 1)
        },
        {
            'ax': axes[4],
            'title': 'Area Ratio',
            'mean_col': 'area_mean',
            'ylim': (0, 1)
        },
        {
            'ax': axes[5],
            'title': 'Intersection Area Ratio',
            'mean_col': 'int_area_mean',
            'ylim': (0, 1)
        },
    ]

    # Plot curves for each model
    for model in models:
        csv_path = _resolve_csv_path(model, param, data_root, data_root)

        if not csv_path:
            raise FileNotFoundError(f"Missing MODE summary: {model}/{param}")

        df = pd.read_csv(csv_path)
        df = df.sort_values('Lead Time (h)')

        x = df['Lead Time (h)'].values
        color = colors_map.get(model, '#333333')
        label = labels_map.get(model, model)

        for sp_idx, sp in enumerate(subplots):
            ax = sp['ax']
            mean_col = sp['mean_col']

            y_mean = df[mean_col].values

            # The selected Appendix B scale has one arm on either side.
            disp_col = mean_col.replace('_mean', '_disp')
            y_lower = df[disp_col].values
            y_upper = df[disp_col].values

            assert np.all(np.isfinite(y_mean))
            assert np.all(np.isfinite(y_lower))
            assert np.all(np.isfinite(y_upper))
            x_plot = x
            y_mean_plot = y_mean
            y_lower_plot = y_lower
            y_upper_plot = y_upper

            # Clip lower bound to ylim minimum (avoid negative error bars for non-negative metrics)
            if sp['ylim'] is not None:
                y_lower_plot = np.minimum(y_lower_plot,
                                          y_mean_plot - sp['ylim'][0])
            yerr = np.vstack([y_lower_plot, y_upper_plot])

            ax.errorbar(
                x_plot,
                y_mean_plot,
                yerr=yerr,
                color=color,
                ecolor=color,
                elinewidth=1.3,
                capsize=3,
                capthick=1.3,
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
    # Remove duplicate legend entries
    unique = dict(zip(labels, handles))
    axes[0].legend(
        unique.values(),
        unique.keys(),
        loc='best',
        fontsize=9,
        framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space for the main title

    # Save image
    save_path = os.path.join(output_dir, f'mode_plots_{param}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_total_interest(
        param,
        models,
        labels_map,
        colors_map,
        data_root,
        output_dir,
        title_suffix):
    """Generate a Total Interest Index line plot for a single parameter"""
    display_name = parameter_display.get(param, param)
    plt.figure(figsize=(12, 7), dpi=100)
    # Remove per-subplot title (avoid redundancy in composite layout)

    for model in models:
        csv_path = _resolve_csv_path(model, param, data_root, data_root)

        if not csv_path:
            raise FileNotFoundError(f"Missing MODE summary: {model}/{param}")

        df = pd.read_csv(csv_path)
        df = df.sort_values('Lead Time (h)')

        x = df['Lead Time (h)'].values
        y_mean = df['interest_mean'].values

        # Use the Appendix B scale stored by MODE_avg.py.
        y_lower = df['interest_disp'].values
        y_upper = df['interest_disp'].values

        assert np.all(np.isfinite(y_mean))
        assert np.all(np.isfinite(y_lower))
        assert np.all(np.isfinite(y_upper))
        x_plot = x
        y_mean_plot = y_mean
        y_lower_plot = y_lower
        y_upper_plot = y_upper

        # Clip lower bound to 0 (interest range [0,1])
        y_lower_plot = np.minimum(y_lower_plot, y_mean_plot)
        yerr = np.vstack([y_lower_plot, y_upper_plot])

        color = colors_map.get(model, '#333333')
        label = labels_map.get(model, model)

        plt.errorbar(
            x_plot,
            y_mean_plot,
            yerr=yerr,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            capthick=1.3,
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
    save_path = os.path.join(output_dir, f'total_interest_{param}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _combine_images(
        image_paths,
        nrows,
        ncols,
        figsize,
        save_path,
        dpi=200,
        titles=None,
        suptitle=None,
        suptitle_y=0.98,
        tight_rect_top=0.95):
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
    # Control distance between main title and subplots: smaller gap means title is closer
    if nrows >= 4:
        suptitle_y = 0.985
        gap = 0.020
    else:
        suptitle_y = 0.990
        gap = 0.018
    return suptitle_y, suptitle_y - gap


# ==================== Main Loop ====================
for group in plot_groups:
    group_name = group['name']
    group_models = group['models']
    group_title = group['title_suffix']
    data_root = group['data_root']

    output_dir = os.path.join(output_base, group_name)
    os.makedirs(output_dir, exist_ok=True)

    for param in parameters:
        plot_parameter(
            param,
            group_models,
            model_labels,
            model_colors,
            data_root,
            output_dir,
            group_title)
        plot_total_interest(
            param,
            group_models,
            model_labels,
            model_colors,
            data_root,
            output_dir,
            group_title)

    # Combine eight figures into one (4x2 and 2x4)
    param_paths = [(p, os.path.join(output_dir, f'mode_plots_{p}.png')) for p in parameters]
    param_paths = [(p, path) for p, path in param_paths if os.path.exists(path)]
    if param_paths:
        combined_paths = [path for _, path in param_paths]
        titles = [
            f"({chr(ord('a') + i)}) {parameter_display.get(p, p)}"
            for i, (p, _) in enumerate(param_paths)
        ]
        suptitle = "Mean MODE of ML-based Models for Tornado Relevant Prameters"

        combined_save_4x2 = os.path.join(output_dir, 'mode_plots_all_4x2.png')
        y_4x2, top_4x2 = _layout_for_grid(4, 2)
        _combine_images(combined_paths, 4, 2, (20, 28),
                        combined_save_4x2, dpi=200, titles=titles, suptitle=suptitle,
                        suptitle_y=y_4x2, tight_rect_top=top_4x2)
        print(f"Saved combined figure: {combined_save_4x2}")

        combined_save_2x4 = os.path.join(output_dir, 'mode_plots_all_2x4.png')
        y_2x4, top_2x4 = _layout_for_grid(2, 4)
        _combine_images(combined_paths, 2, 4, (28, 16),
                        combined_save_2x4, dpi=200, titles=titles, suptitle=suptitle,
                        suptitle_y=y_2x4, tight_rect_top=top_2x4)
        print(f"Saved combined figure: {combined_save_2x4}")

    # Combine Total Interest Index figures (4x2 and 2x4)
    interest_param_paths = [(p, os.path.join(output_dir, f'total_interest_{p}.png')) for p in parameters]
    interest_param_paths = [(p, path) for p, path in interest_param_paths if os.path.exists(path)]
    if interest_param_paths:
        interest_paths = [path for _, path in interest_param_paths]
        interest_titles = [
            f"({chr(ord('a') + i)}) {parameter_display.get(p, p)}"
            for i, (p, _) in enumerate(interest_param_paths)
        ]
        suptitle = "Mean MODE of ML-based Models for Tornado Relevant Prameters"

        interest_save_4x2 = os.path.join(
            output_dir, 'total_interest_all_4x2.png')
        y_4x2, top_4x2 = _layout_for_grid(4, 2)
        _combine_images(interest_paths, 4, 2, (18, 26),
                        interest_save_4x2, dpi=200, titles=interest_titles, suptitle=suptitle,
                        suptitle_y=y_4x2, tight_rect_top=top_4x2)
        print(f"Saved combined figure: {interest_save_4x2}")

        interest_save_2x4 = os.path.join(
            output_dir, 'total_interest_all_2x4.png')
        y_2x4, top_2x4 = _layout_for_grid(2, 4)
        _combine_images(interest_paths, 2, 4, (26, 14),
                        interest_save_2x4, dpi=200, titles=interest_titles, suptitle=suptitle,
                        suptitle_y=y_2x4, tight_rect_top=top_2x4)
        print(f"Saved combined figure: {interest_save_2x4}")

print("All parameter plots completed!")
