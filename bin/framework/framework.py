import os
import logging
import pandas as pd
from collections import defaultdict
from nltk.stem import Cistem
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from tqdm import tqdm

from bin.framework.example_data import EXAMPLES
from bin.framework.metrics import *  # noqa: F403
from bin.framework.metrics import Metric
from bin.utils.logger import TqdmLoggingHandler

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class Framework:
    """
    A class to represent a framework for evaluating text generation models.
    """

    def __init__(self, metrics: Optional[List[Metric]] = None):
        self.cistem = Cistem()
        self._initialize_models()
        self._initialize_logger()

        self.hypotheses_metrics = [
            Metric.DISTINCT_1,
            Metric.DISTINCT_2,
            Metric.TTR,
            Metric.MOVING_AVERAGE_TTR,
            Metric.AVERAGE_N_OF_TOKENS,
            Metric.AVERAGE_N_OF_CHARACTERS,
            Metric.DISTANCE_TO_CENTROID,
            Metric.DISCOURSE_COHERENCE,
            Metric.INTER_SENTENCE_SIMILARITY,
            Metric.POS_TAG_N_GRAMS_DIVERSITY,
        ]
        self.comparison_metrics = [
            Metric.BLEU,
            Metric.MEAN_LEVENSHTEIN_DISTANCE,
        ]

        if metrics is None:
            self.metrics = self.hypotheses_metrics + self.comparison_metrics
        else:
            self.metrics = metrics

    def _initialize_models(self):
        """Initialize models and tokenizers."""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.g_model = AutoModelForMaskedLM.from_pretrained(
            "google-bert/bert-base-german-cased"
        )
        self.g_tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-german-cased"
        )

    def _initialize_logger(self):
        """Initialize the logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(TqdmLoggingHandler())

    def compute_hypotheses_metrics(self, hypotheses: List[str]) -> dict:
        """Compute metrics for a list of hypotheses."""
        results = defaultdict(float)
        for metric in self.metrics:
            if metric in self.hypotheses_metrics:
                self._compute_metric(metric, hypotheses, results)
        return dict(results)

    def compute_comparison_metrics(
        self, references: List[str], hypotheses: List[str]
    ) -> dict:
        """Compute metrics for a list of references and hypotheses."""
        results = defaultdict(float)
        for metric in self.metrics:
            if metric in self.comparison_metrics:
                self._compute_metric(metric, references, results, hypotheses)
        return dict(results)

    def _compute_metric(
        self,
        metric: Metric,
        data: List[str],
        results: dict,
        hypotheses: Optional[List[str]] = None,
    ):
        """Compute a single metric and log the result."""
        metric_name = metric.name.lower()
        if metric == Metric.DISTINCT_1:
            results[metric_name] = round(distinct_n(data, 1), 3)
        elif metric == Metric.DISTINCT_2:
            results[metric_name] = round(distinct_n(data, 2), 3)
        elif metric == Metric.TTR:
            results[metric_name] = round(type_token_ratio(data), 3)
        elif metric == Metric.MOVING_AVERAGE_TTR:
            results[metric_name] = round(moving_average_ttr(data), 3)
        elif metric == Metric.AVERAGE_N_OF_TOKENS:
            results[metric_name] = round(average_n_of_tokens(data), 3)
        elif metric == Metric.AVERAGE_N_OF_CHARACTERS:
            results[metric_name] = round(average_n_of_characters(data), 3)
        elif metric == Metric.DISTANCE_TO_CENTROID:
            results[metric_name] = round(
                distance_to_centroid(data, model=self.model), 3
            )
        elif metric == Metric.DISCOURSE_COHERENCE:
            results[metric_name] = round(discourse_coherence(data), 3)
        elif metric == Metric.INTER_SENTENCE_SIMILARITY:
            results[metric_name] = round(
                inter_sentence_similarity(data, model=self.model), 3
            )
        elif metric == Metric.BLEU:
            results[metric_name] = round(bleu_score(hypotheses, data), 3)
        elif metric == Metric.MEAN_LEVENSHTEIN_DISTANCE:
            results[metric_name] = round(mean_levenshtein_distance(data, hypotheses), 3)
        elif metric == Metric.POS_TAG_N_GRAMS_DIVERSITY:
            results[metric_name] = round(
                pos_tag_n_grams_diversity(data, hypotheses, 2), 3
            )
        self.logger.info(f"{metric_name}: {results[metric_name]}")

    def __apply_framework(
        self,
        references: Optional[List[str]] = None,
        hypotheses: Optional[List[str]] = None,
    ) -> dict:
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        results = {}
        self.logger.info("Computing metrics for hypotheses.")
        results.update(self.compute_hypotheses_metrics(hypotheses))
        if references:
            self.logger.info("Computing comparison metrics.")
            results.update(self.compute_comparison_metrics(references, hypotheses))
        self.logger.info(f"Results: {results}")
        return {"results": results}

    def apply_framework(self, data: List[dict]) -> List[dict]:
        """
        Apply the framework to a text generation model.
        Returns:
            List[dict]: The results of the evaluation.
        """
        results = []
        for item in data:
            reference = item.get("reference")
            hypothesis = item.get("hypothesis")
            result = self.__apply_framework(reference, hypothesis)
            results.append(result)
        return results

    def apply_framework_to_datasets(
        self, dataset_a: pd.DataFrame, dataset_b: Optional[pd.DataFrame] = None
    ) -> List[dict]:
        """
        Apply the framework to a text generation model.
        Returns:
            List[dict]: The results of the evaluation.
        """
        intents = [intent for intent in dataset_a.intent.unique() if intent is not None]
        self.logger.info(f"Evaluating intents: {intents}")
        results = []

        for intent in tqdm(intents):
            self.logger.info(f"Evaluating intent: {intent}")
            hypotheses = dataset_a[dataset_a.intent == intent].text.tolist()
            references = (
                dataset_b[dataset_b.intent == intent].text.tolist()
                if dataset_b is not None
                else None
            )

            if not hypotheses:
                self.logger.warning(
                    f"No data found for intent {intent}. Skipping evaluation."
                )
                continue

            result = self.__apply_framework(references, hypotheses)
            results.append({intent: result})

        return results


if __name__ == "__main__":
    framework = Framework()
    results = framework.apply_framework(EXAMPLES)
    print(results)
