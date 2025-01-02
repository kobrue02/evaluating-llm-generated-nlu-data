import json
import pandas as pd
import numpy as np

from bin.framework.framework import Framework


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


df = load_data_full()

framework = Framework()
results = framework.apply_framework_to_datasets(df, df.copy())

for result in results:
    print(result)

mean_metrics = calculate_mean_metrics(results)
print(mean_metrics)
