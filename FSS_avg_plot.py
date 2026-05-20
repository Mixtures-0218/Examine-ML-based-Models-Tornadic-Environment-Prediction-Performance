import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Output path
output_base = os.path.join('avg_fss')
os.makedirs(output_base, exist_ok=True)

# Model list
models_ai = ['pangu', 'fengwu_v1', 'fengwu_v2', 'fuxi']
parameters = [
    'cape',
    'cin',
    'k_idx',
    'positive_helicity',
    'negative_helicity',
    'lcl',
    'lfc',
    'ehi']

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

# Color scheme
model_colors = {
    'pangu': '#1f77b4',
    'fengwu_v1': '#2e8b57',
    'fengwu_v2': '#3cb371',
    'fuxi': '#d62728',
}

model_labels = {
    'pangu': 'Pangu',
    'fengwu_v1': 'FengWu v1',
    'fengwu_v2': 'FengWu v2',
    'fuxi': 'FuXi',
}

lead_times = [24, 48, 72, 120, 168]

# Plot groups: AI only (10 times)



plot_groups = [
    {
        'name': 'ai_10',
        'models': models_ai,
        'title_suffix': 'AI (10 times)',
        'data_root': 'avg_scores'
    }
]

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


def plot_group(group):
    group_name = group['name']
    group_models = group['models']
    group_title = group['title_suffix']
    data_root = group['data_root']

    output_dir = os.path.join(output_base, group_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create a separate figure for each parameter
    for parameter in parameters:
        # Create new figure (critical fix: new figure per parameter loop)
        plt.figure(figsize=(12, 7), dpi=100)

        print(f"\nProcessing parameter: {parameter} | {group_title}")

        # Loop over each model and plot
        for model in group_models:
            try:
                # Load data
                data_path = _resolve_data_path(
                    model, parameter, data_root, data_root)

                if not data_path:
                    raise FileNotFoundError

                data = np.load(data_path)

                mean_scores = data['mean']
                fss_mean = mean_scores[:, 2]

                # Asymmetric error bar: prefer lower_err/upper_err, fallback to symmetric dispersion
                if 'lower_err' in data and 'upper_err' in data:
                    fss_lower = data['lower_err'][:, 2]
                    fss_upper = data['upper_err'][:, 2]
                    fss_yerr = np.vstack([fss_lower, fss_upper])
                else:
                    std_scores = data['dispersion']
                    fss_std = std_scores[:, 2]
                    fss_yerr = fss_std

                color = model_colors[model]
                label = model_labels.get(model, model)

                # Plot mean line with asymmetric error bars
                plt.errorbar(
                    lead_times,
                    fss_mean,
                    yerr=fss_yerr,
                    color=color,
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=4,
                    capthick=1.6,
                    linewidth=2.5,
                    label=label,
                    marker='o',
                    markersize=6,
                    markerfacecolor=color,
                    markeredgecolor='white',
                    markeredgewidth=1)

                print(f"  Successfully loaded model: {model}")

            except FileNotFoundError:
                print(f"  Warning: Data file for model {model} not found, skipping")
            except Exception as e:
                print(f"  Warning: Error loading model {model}: {e}, skipping")

        # Add grid lines
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Set axis labels and title
        display_name = parameter_display.get(parameter, parameter)
        plt.xlabel('Lead Time (hours)', fontsize=13, fontweight='bold')
        plt.ylabel('FSS Score', fontsize=13, fontweight='bold')
        # Remove per-subplot title (avoid redundancy in composite layout)

        # Set x-axis ticks
        plt.xticks(lead_times, [f'{h}h' for h in lead_times], fontsize=11)
        plt.yticks(fontsize=11)

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(
            unique.values(),
            unique.keys(),
            fontsize=11,
            framealpha=0.95,
            loc='best',
            ncol=2)

        # Set y-axis range
        plt.ylim(0, 1)

        # Auto adjust layout
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(output_dir, f'fss_avg_{parameter}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

        # Close current figure to free memory
        plt.close()


def _combine_images(
        image_paths,
        nrows,
        ncols,
        figsize,
        save_path,
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
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _layout_for_grid(nrows, ncols):
    # Control the distance between the main title and subplots: smaller gap means title is closer to subplots
    if nrows >= 4:
        suptitle_y = 0.985
        gap = 0.020
    else:
        suptitle_y = 0.990
        gap = 0.018
    return suptitle_y, suptitle_y - gap


def combine_group_images(output_dir):
    param_paths = [
        (p, os.path.join(output_dir, f'fss_avg_{p}.png')) for p in parameters
    ]
    param_paths = [(p, path) for p, path in param_paths if os.path.exists(path)]
    if param_paths:
        combined_paths = [path for _, path in param_paths]
        titles = [
            f"({chr(ord('a') + i)}) {parameter_display.get(p, p)}"
            for i, (p, _) in enumerate(param_paths)
        ]
        suptitle = "Mean FSS of ML-based Models for Tornado-Relevant Parameters"

        combined_save_4x2 = os.path.join(output_dir, 'fss_avg_all_4x2.png')
        y_4x2, top_4x2 = _layout_for_grid(4, 2)
        _combine_images(
            combined_paths, 4, 2, (18, 26), combined_save_4x2,
            titles=titles, suptitle=suptitle,
            suptitle_y=y_4x2, tight_rect_top=top_4x2)
        print(f"Saved combined figure: {combined_save_4x2}")

        combined_save_2x4 = os.path.join(output_dir, 'fss_avg_all_2x4.png')
        y_2x4, top_2x4 = _layout_for_grid(2, 4)
        _combine_images(
            combined_paths, 2, 4, (26, 14), combined_save_2x4,
            titles=titles, suptitle=suptitle,
            suptitle_y=y_2x4, tight_rect_top=top_2x4)
        print(f"Saved combined figure: {combined_save_2x4}")


for group in plot_groups:
    plot_group(group)
    combine_group_images(os.path.join(output_base, group['name']))

print("\nAll figures processed!")
