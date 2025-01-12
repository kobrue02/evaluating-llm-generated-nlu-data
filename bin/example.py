import json
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from bin.framework.framework import Framework


logging.basicConfig(level=logging.INFO)


def load_ubuntucorpus():
    json_data = json.loads(open("AskUbuntuCorpus.json", encoding="utf-8").read())
    df = pd.DataFrame(json_data["sentences"])

    df = df[["text", "intent"]]

    # for each intent, keep 5 random samples
    df = df.groupby("intent").apply(lambda x: x.sample(n=5)).reset_index(drop=True)

    return df


def load_data_full():
    json_data = json.loads(open("data_full.json", encoding="utf-8").read())
    train_data = json_data["train"]
    texts = [item[0] for item in train_data]
    intents = [item[1] for item in train_data]
    df = pd.DataFrame({"text": texts, "intent": intents})
    # for each intent, keep 10 random samples
    df = df.groupby("intent").apply(lambda x: x.sample(n=10)).reset_index(drop=True)
    return df


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


def plot_results_df(df: pd.DataFrame):
    # make a dashboard of plots
    # where each plot is the distribution of a metric
    # across all intents
    n_metrics = len(df.columns) - 1
    n_cols = 2
    n_rows = n_metrics // n_cols + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for i, metric in enumerate(df.columns[1:]):
        if metric == "intent":
            continue
        ax = axs[i]
        ax.hist(df[metric], bins=20)
        # add a vertical line for the mean
        ax.axvline(df[metric].mean(), color="red", linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

df = load_data_full()

framework = Framework()
results = framework.apply_framework_to_datasets(
    df, df.copy().sample(frac=1).reset_index(drop=True)
)

print(results)

results_df = results_to_dataframe(results)

print(results_df)

plot_results_df(results_df)
