from bin.utils.read_datasets import read_sipgate_dataset
from bin.framework.framework import Framework

import logging

logging.basicConfig(level=logging.INFO)


def evaluate_sipgate_dataset(queries_per_intent: int = 10, subset: int = None):
    sipgate_data = read_sipgate_dataset()
    if queries_per_intent:
        sipgate_data = sipgate_data.groupby("intent").head(queries_per_intent)
    if subset:
        sipgate_data = sipgate_data.sample(subset)

    framework = Framework()
    result = framework.apply_framework_to_datasets(sipgate_data)
    for intent in result:
        print(intent)


if __name__ == "__main__":
    evaluate_sipgate_dataset(10)
