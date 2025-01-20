import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from bin.framework.framework import Framework


logging.basicConfig(level=logging.WARN)


def load_ubuntucorpus():
    json_data = json.loads(open("AskUbuntuCorpus.json", encoding="utf-8").read())
    df = pd.DataFrame(json_data["sentences"])

    df = df[["text", "intent"]]

    # for each intent, keep 5 random samples
    df = df.groupby("intent").apply(lambda x: x.sample(n=5)).reset_index(drop=True)

    return df


def load_data_full(n: int = None):
    json_data = json.loads(open("data_full.json", encoding="utf-8").read())
    train_data = json_data["train"]
    texts = [item[0] for item in train_data]
    intents = [item[1] for item in train_data]
    df = pd.DataFrame({"text": texts, "intent": intents})
    # for each intent, keep 10 random samples
    df = df.groupby("intent").apply(lambda x: x.sample(n=10)).reset_index(drop=True)
    if n:
        # keep only n intents
        intents = df["intent"].unique()[:n]
        df = df[df["intent"].isin(intents)]
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


df = load_data_full()
df_golden = df.copy().sample(frac=1).reset_index(drop=True)
df_generated = df.copy().sample(frac=1).reset_index(drop=True)

framework = Framework()
results = framework.apply_framework_to_datasets(
    golden_data=df_golden,
    generated_data=df_generated,
)

print(results)

results_df = results_to_dataframe(results)

print(results_df)

plot_results_df(results_df)
