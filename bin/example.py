import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

from bin.framework.framework import Framework
from bin.utils.read_datasets import read_sipgate_dataset

logging.basicConfig(level=logging.INFO)


def calculate_mean_metrics(results):
    # Initialize dictionaries to store sums and counts
    metric_sums = {}
    metric_counts = {}

    # Process each intent dictionary
    for intent_dict in results:
        for intent_name, data in intent_dict.items():
            results = data["results"]
            for metric, value in results.items():
                if np.isnan(value):
                    continue
                metric_sums[metric] = metric_sums.get(metric, 0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

    # Calculate means
    mean_metrics = {
        metric: round(metric_sums[metric] / metric_counts[metric], 2)
        for metric in metric_sums
    }

    return mean_metrics


def results_to_dataframe(results: list[dict]):
    # Initialize an empty DataFrame
    columns = ["intent"] + list(list(results[0].values())[0]["results"].keys())
    df = pd.DataFrame(columns=columns)
    df.set_index("intent", inplace=True)

    # Process each intent dictionary
    for intent_dict in results:
        for intent_name, data in intent_dict.items():
            results = data["results"]
            df.loc[intent_name] = results

    df["intent"] = df.index
    df.reset_index(drop=True, inplace=True)

    return df


def plot_results_df(df: pd.DataFrame, plot_func=plt.hist):
    # make a dashboard of plots
    # where each plot is the distribution of a metric
    # across all intents
    n_metrics = len([col for col in df.columns if not df[col].isnull().all()]) - 1
    n_cols = 2
    n_rows = n_metrics // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for i, metric in enumerate(df.columns[1:]):
        if metric == "intent" or df[metric].isnull().all():
            continue

        x, y = df[metric].index, df[metric].values

        ax = axs[i]
        sns.histplot(
            y,
            ax=ax,
            kde=True,
            bins=20,
            color="blue",
            alpha=0.5,
            linewidth=0,
        )
        ax.set_title(metric)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


df = read_sipgate_dataset()
print(df.head())

framework = Framework()
results = framework.apply_framework_to_datasets(df)

print(results)
results_df = results_to_dataframe(results)

print(results_df)
plot_results_df(results_df)

