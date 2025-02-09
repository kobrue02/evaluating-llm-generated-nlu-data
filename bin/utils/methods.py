import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from bin.framework.framework import Framework


def catplot_metric_dfs(metric_dfs: list[pd.DataFrame], metric_names: list[str]):
    n_metrics = len(metric_dfs)
    n_cols = 2
    n_rows = -(-n_metrics // n_cols)  # Ceiling division
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axs in case of single row
    axs = axs.flatten() if n_metrics > 1 else [axs]

    for i, metric_df in enumerate(metric_dfs):
        metric_df_melted = metric_df.melt(id_vars=['intent'], var_name='Metric', value_name='Value')
        metric_df_melted = remove_outliers(metric_df_melted, "Value")
        sns.boxplot(
            data=metric_df_melted,
            x='Metric',
            y='Value',
            ax=axs[i],
            palette="pastel",
            hue="Metric",
            legend=False
            )
        # rotate x labels
        axs[i].get_xaxis().set_tick_params(rotation=90)
        axs[i].set_title(metric_names[i].value)
    plt.tight_layout()
    plt.show()

def remove_outliers(df: pd.DataFrame, metric: str):
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
    return df

def load_df(dataset_name: str):
    gen_df = pd.read_csv(f"data/{dataset_name}.csv", encoding="utf-8")
    gen_df.rename(columns={"query": "text"}, inplace=True)
    gen_df = gen_df[gen_df["text"].apply(lambda x: isinstance(x, str))]
    # remove any row that contains "note" or "these queries"
    gen_df = gen_df[~gen_df["text"].str.contains("note|these queries|here are|Here are", case=False)]
    # remove any row that has an empty text
    gen_df = gen_df[gen_df["text"].apply(lambda x: len(x) > 0)]
    gen_df.reset_index(drop=True, inplace=True)
    return gen_df

def transform_dfs_to_metric_dfs(dfs: list[pd.DataFrame], dataset_names:list[str], columns=None):
    metric_dfs = []
    metrics = [
        col for col in dfs[0].columns
        if not dfs[0][col].isnull().all()
        and col != "intent"
        ]
    for metric in metrics:
      if not columns:
        columns = ["intent"] + dataset_names
      df = pd.DataFrame(columns=columns)
      df.set_index("intent", inplace=True)
      for i, df_ in enumerate(dfs):
        df[columns[i+1]] = df_[metric]
      df["intent"] = df.index
      df.reset_index(drop=True, inplace=True)
      metric_dfs.append(df)
    return metric_dfs

def dfs_to_stripplots(dfs: list[pd.DataFrame]):
    # make a dashboard of plots
    # where each plot is the distribution of a metric
    # across all intents
    n_metrics = len([col for col in dfs[0].columns if not dfs[0][col].isnull().all()]) - 1
    n_cols = 2
    n_rows = -(-n_metrics // n_cols)  # Ceiling division

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()
    colors = sns.color_palette("husl", n_rows)
    for i, metric in enumerate(dfs[0].columns[1:]):
        if metric == "intent":
            continue

        ax = axs[i]
        ax.set_title(metric)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        for i, df in enumerate(dfs):
          x, y = df[metric].index, df[metric].values
          sns.stripplot(
            data=df,
            x=metric,
            ax=ax,
            color=colors[i],
            alpha=0.5,
            linewidth=0,
          )

    plt.tight_layout()
    plt.show()

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

def read_sipgate_dataset() -> pd.DataFrame:
    """
    Read the sipgate dataset and return a DataFrame with the specified columns.
    If columns is None, return the full DataFrame.
    """
    dataset = pd.read_csv("data/sipgate_data.csv")
    dataset.rename(columns={"phraseIntent": "intent"}, inplace=True)
    dataset = dataset[
        ["text", "intent", "occurrences", "phraseEntTypes", "annotations"]
    ]
    return dataset

def clean_sipgate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sipgate dataset by removing duplicates, intents with too few samples and samples with low occurrences.
    """

    # deduplicate text column
    df = df.drop_duplicates(subset="text")

    # keep only intents with more than 20 and less than 1000 samples
    df = df.groupby("intent").filter(lambda x: len(x) > 100 and len(x) < 500)

    # keep only intents that don't start with "_"
    df = df[~df.intent.str.startswith("_")]

    # for each intent, keep the 100 samples with highest value in occurrences column
    df = df.groupby("intent").apply(lambda x: x.nlargest(25, "occurrences"))

    df.reset_index(drop=True, inplace=True)
    return df

def load_sipgate_dataset():
    df = read_sipgate_dataset()
    df = clean_sipgate_dataset(df)
    return df

def clean_synthetic_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # remove any row that contains "note" or "these queries"
    df = df[~df["text"].str.contains("note|these queries|here are|Here are|json", case=False)]
    # remove any row that has an empty text
    df = df[df["text"].apply(lambda x: len(x) > 0)]
    df.reset_index(drop=True, inplace=True)
    return df