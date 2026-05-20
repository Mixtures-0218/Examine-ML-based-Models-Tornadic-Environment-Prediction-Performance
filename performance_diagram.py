import os
import matplotlib.pyplot as plt
import numpy as np


def performance_diagram(lead_times, scores):
    """
    Optimized Performance Diagram
    scores expected shape (N, 4), column order [POD, CSI, FSS, SR]
    """
    # Extract data: trans_score[0] is POD, trans_score[3] is SR
    trans_score = scores.transpose(1, 0)
    pod = trans_score[0]
    sr = trans_score[3]

    fig, ax = plt.subplots(figsize=(9, 8))

    # 1. Plot CSI contours with shaded layers
    x_grid = np.linspace(0.01, 1, 100)
    y_grid = np.linspace(0.01, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x_grid, y_grid)
    CSI_grid = 1 / (1 / SR_grid + 1 / POD_grid - 1)

    # Light gray fill for visual depth
    ax.contourf(
        SR_grid,
        POD_grid,
        CSI_grid,
        levels=np.arange(
            0,
            1.1,
            0.1),
        cmap='Greys',
        alpha=0.15)
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
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f', inline_spacing=5)

    # 2. Optimized Bias lines with dynamically placed labels
    bias_levels = [0.1, 0.3, 0.5, 0.8, 1, 1.25, 1.5, 2, 3, 5, 10]
    for b in bias_levels:
        ls = '-' if b == 1 else ':'
        alpha = 0.6 if b == 1 else 0.3
        # Compute Bias line
        x_bias = np.array([0, 1])
        y_bias = b * x_bias
        ax.plot(
            x_bias,
            y_bias,
            color='black',
            linestyle=ls,
            alpha=alpha,
            zorder=1)

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

    # 3. Lead Time evolution trajectory (core optimization)
    # Dashed line connecting points to show performance drift with increasing lead time
    ax.plot(
        sr,
        pod,
        color='teal',
        alpha=0.4,
        linestyle='-',
        linewidth=1.5,
        zorder=2)

    # 4. Plot data points
    scatter = ax.scatter(
        sr,
        pod,
        c=range(
            len(lead_times)),
        cmap='viridis',
        s=100,
        edgecolors='white',
        linewidths=1.5,
        zorder=3,
        label='Lead Times')

    # Annotate each point with offset to avoid overlap
    for i, lead in enumerate(lead_times):
        ax.annotate(
            lead,
            (sr[i],
             pod[i]),
            xytext=(
                5,
                5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='darkslategray')

    # 5. Detail refinements
    ax.set_aspect('equal')  # Critical: Performance Diagram must be square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=12, fontweight='bold')
    ax.set_ylabel(
        "Probability of Detection (POD)",
        fontsize=12,
        fontweight='bold')
    ax.set_title(
        f"Performance: {model} | {parameter} 75th percentile",
        fontsize=14,
        pad=20)

    # Add grid lines at the bottom layer
    ax.grid(True, which='both', linestyle=':', alpha=0.2, zorder=0)

    # Colorbar indicating Lead Time sequence
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.1)
    cbar.set_label('Lead Time Sequence', rotation=270, labelpad=15)
    cbar.set_ticks([])  # Hide specific values, text annotations suffice

    plt.tight_layout()
    plt.show()


parameter = 'cape'

time = '2024-05-26-04UTC'
model = 'fengwu_v2'
if model != 'gfs':
    lead_times = ['24h', '48h', '72h', '120h', '168h']
    scores = np.load(
        os.path.join(
            time,
            model,
            'scores',
            f'scores_{parameter}_{model}.npy'))
else:
    lead_times = []
    for file in os.listdir(os.path.join(time, model)):
        if file.endswith('.nc') and '._' not in file:
            lead_times.append(file[-6:-3])
    lead_times.sort()
    lead_times = lead_times[1:]
    scores = np.load(
        os.path.join(
            time,
            model,
            'scores',
            f'scores_{parameter}_{model}.npy'))[
        1:]


# scores = np.load('scores_mean_cape_pangu.npy')
performance_diagram(lead_times, scores)
