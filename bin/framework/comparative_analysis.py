import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def set_acl_style():
    """Set the plotting style to match ACL paper aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = ["#2878B5", "#9AC9DC", "#C82423", "#F8AC8C", "#6F4C9B", "#FFBE7A"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "patch.linewidth": 1.0,
            "patch.edgecolor": "black",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.prop_cycle": plt.cycler("color", colors),
        }
    )


def compare_strategies(results, save_path=None):
    """
    Compare different model strategies with ACL-style plots.

    Args:
        results (dict): Dictionary of model results
        save_path (str, optional): Path to save the figure
    """
    # Set ACL style
    set_acl_style()

    # Restructure the data for plotting
    df_data = {
        metric: [model_results[metric] for model_results in results.values()]
        for metric in next(iter(results.values())).keys()
    }
    df = pd.DataFrame(df_data, index=results.keys())

    # Create subplots with ACL-friendly size
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Define metrics and their y-axis ranges
    metric_configs = {
        "perplexity": {"title": "Perplexity", "ylabel": "Score (lower is better)"},
        "diversity": {"title": "Diversity", "ylabel": "Score"},
        "coherence": {"title": "Coherence", "ylabel": "Score"},
        # 'task_performance': {'title': 'Task Performance', 'ylabel': 'Score'}
    }

    for i, (metric, config) in enumerate(metric_configs.items()):
        ax = axes[i // 2, i % 2]

        # Create bar plot
        bars = df[metric].plot(kind="bar", ax=ax, width=0.7)

        # Customize each subplot
        ax.set_title(config["title"], pad=10)
        ax.set_xlabel("Model", labelpad=10)
        ax.set_ylabel(config["ylabel"], labelpad=10)

        # Add value labels on top of bars
        for j, v in enumerate(df[metric]):
            ax.text(j, v, f"{v:.2f}", ha="center", va="bottom")

        # Customize grid
        ax.grid(True, linestyle="--", axis="y", alpha=0.7)
        ax.set_axisbelow(True)

        # Remove unnecessary spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust tick labels
        ax.tick_params(axis="x", rotation=45)

        # Set y-axis to start from 0 for scores
        if metric != "perplexity":
            ax.set_ylim(0, 1.1)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, axes


if __name__ == "__main__":
    # Example results
    llama_results = {
        "perplexity": 30.1,
        "diversity": 0.5,
        "coherence": 0.8,
        "task_performance": 0.9,
    }
    phi_results = {
        "perplexity": 35.2,
        "diversity": 0.4,
        "coherence": 0.7,
        "task_performance": 0.85,
    }

    # Create comparison plots and save
    fig, axes = compare_strategies(
        results={"Llama": llama_results, "Phi": phi_results},
        save_path="model_comparison.pdf",
    )
