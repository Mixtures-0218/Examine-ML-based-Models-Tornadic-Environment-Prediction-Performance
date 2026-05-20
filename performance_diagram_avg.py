import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os


def performance_diagram_with_ellipse(
        lead_times,
        mean_scores,
        std_scores,
        model,
        parameter,
        sigma=1):
    # mean_scores/std_scores: (5,4)  column order [POD, CSI, FSS, SR]
    pod = mean_scores[:, 0]
    sr = mean_scores[:, 3]
    pod_std = std_scores[:, 0]
    sr_std = std_scores[:, 3]

    fig, ax = plt.subplots(figsize=(9, 8))

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

    n_points = len(lead_times)
    colors = plt.cm.plasma(np.linspace(0, 1, n_points))

    # Mean trajectory and points - use gradient colors
    for i in range(n_points - 1):
        ax.plot(sr[i:i + 2], pod[i:i + 2],
                color=colors[i], alpha=0.5, linewidth=1.5)

    # Plot the last segment
    if n_points > 1:
        ax.plot(sr[-2:], pod[-2:],
                color=colors[-2], alpha=0.5, linewidth=1.5)

    # Plot points - each point uses a different color
    scatter = ax.scatter(sr, pod, s=80,
                         c=range(n_points),  # Use indices as color values
                         cmap='plasma',
                         edgecolors='white',
                         linewidths=1.2,
                         zorder=3)

    # Ellipses (1 sigma) - each ellipse uses the corresponding point's color but lighter
    for i in range(len(lead_times)):
        e = Ellipse(
            (sr[i], pod[i]),
            width=2 * sigma * sr_std[i],
            height=2 * sigma * pod_std[i],
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.20,
            zorder=2
        )
        ax.add_patch(e)

        # Add annotation
        ax.annotate(lead_times[i], (sr[i], pod[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    color='darkslategray')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.1)
    cbar.set_label('Lead Time Sequence', rotation=270, labelpad=15)
    cbar.set_ticks([])

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=12, fontweight='bold')
    ax.set_ylabel(
        "Probability of Detection (POD)",
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f"Average Performance: {model} | {parameter} 75th percentile",
        fontsize=14,
        pad=20)
    ax.grid(True, linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.show()


model = 'fuxi'
parameter = 'lfc'
data = np.load(
    os.path.join(
        'avg_scores',
        model,
        f'scores_meanvar_{parameter}_{model}.npz'))
mean_scores = data['mean']
std_scores = data['dispersion']
print(std_scores)

lead_times = ['24h', '48h', '72h', '120h', '168h']
performance_diagram_with_ellipse(
    lead_times,
    mean_scores,
    std_scores,
    model,
    parameter)
