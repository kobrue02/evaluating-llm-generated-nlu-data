import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


import logging
import pandas as pd

from collections import defaultdict
from nltk.stem import Cistem
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bin.framework.example_data import EXAMPLES
from bin.framework.metrics import *  # noqa: F403
from bin.framework.metrics import Metric
from bin.utils.logger import TqdmLoggingHandler

logging.getLogger("transformers").setLevel(logging.ERROR)


class Framework:
    """
    A class to represent a framework for evaluating text generation models.
    Attributes:
        cistem (Cistem): The Cistem stemmer.
    Methods:
        calculate_perplexity(text, model, tokenizer): Calculate the perplexity of a given text using BERT's masked language modeling.
        distinct_n(text, n): Calculate the distinct-n metric of a given text.
        calculate_coherence(text, model): Calculate the coherence of a given text using a sentence embedding model.
        type_token_ratio(text): Calculate the type-token ratio of a given text using the Cistem stemmer.
        moving_average_ttr(text, window_size): Calculate the moving average type-token ratio of a given text using the Cistem stemmer.
        bleu_score(hypothesis, reference): Calculate the BLEU score of a given hypothesis with respect to a reference.
        task_specific_performance(train_data, test_data, model): Calculate the performance of a model on a task-specific dataset.
        inter_sentence_similarity(sentences, model): Calculate the inter-sentence similarity of a list of sentences using a sentence embedding model.
        discourse_coherence(text, model): Calculate the discourse coherence of a given text using the entity grid model.
        apply_framework(): Calculate all metrics for a text generation model.
    """

    def __init__(self, metrics: list[Metric] = None):
        self.cistem = Cistem()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.g_model = AutoModelForMaskedLM.from_pretrained(
            "google-bert/bert-base-german-cased"
        )
        self.g_tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-german-cased"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(TqdmLoggingHandler())

        if metrics is None:
            self.metrics = [
                Metric.PERPLEXITY,
                Metric.DISTINCT_1,
                Metric.DISTINCT_2,
                Metric.TTR,
                Metric.MOVING_AVERAGE_TTR,
                Metric.AVERAGE_N_OF_TOKENS,
                Metric.AVERAGE_N_OF_CHARACTERS,
                Metric.DISTANCE_TO_CENTROID,
                Metric.DISCOURSE_COHERENCE,
                Metric.INTER_SENTENCE_SIMILARITY,
                Metric.BLEU,
                Metric.MEAN_LEVENSHTEIN_DISTANCE,
                Metric.POS_TAG_N_GRAMS_DIVERSITY,
            ]
        else:
            self.metrics = metrics

    def compute_hypotheses_metrics(self, hypotheses) -> dict:
        """Compute metrics for a list of hypotheses."""
        results = defaultdict(float)

        if Metric.PERPLEXITY in self.metrics:
            # Calculate perplexity
            perplexity = calculate_perplexity(
                hypotheses, model=self.g_model, tokenizer=self.g_tokenizer
            )
            results["perplexity"] = round(perplexity, 3)
            self.logger.info(f"Perplexity: {perplexity}")

        if Metric.DISTINCT_1 in self.metrics:
            # Calculate distinct-1
            distinct_1 = distinct_n(hypotheses, 1)
            results["distinct_1"] = round(distinct_1, 3)
            self.logger.info(f"Distinct-1: {distinct_1}")

        if Metric.DISTINCT_2 in self.metrics:
            # Calculate distinct-2
            distinct_2 = distinct_n(hypotheses, 2)
            results["distinct_2"] = round(distinct_2, 3)
            self.logger.info(f"Distinct-2: {distinct_2}")

        if Metric.TTR in self.metrics:
            # Calculate type-token ratio
            ttr = type_token_ratio(hypotheses)
            results["ttr"] = round(ttr, 3)
            self.logger.info(f"TTR: {ttr}")

        if Metric.MOVING_AVERAGE_TTR in self.metrics:
            # Calculate moving average TTR
            ma_ttr = moving_average_ttr(hypotheses)
            results["moving_average_ttr"] = round(ma_ttr, 3)
            self.logger.info(f"Moving average TTR: {ma_ttr}")

        if Metric.AVERAGE_N_OF_TOKENS in self.metrics:
            # Average number of tokens
            average_n_tokens = average_n_of_tokens(hypotheses)
            results["average_n_of_tokens"] = round(average_n_tokens, 3)
            self.logger.info(f"Average number of tokens: {average_n_tokens}")

        if Metric.AVERAGE_N_OF_CHARACTERS in self.metrics:
            # Average number of characters
            average_n_characters = average_n_of_characters(hypotheses)
            results["average_n_of_characters"] = round(average_n_characters, 3)
            self.logger.info(f"Average number of characters: {average_n_characters}")

        if Metric.DISTANCE_TO_CENTROID in self.metrics:
            # Mean distance to centroid
            centroid_distance = distance_to_centroid(hypotheses, model=self.model)
            results["centroid_distance"] = round(centroid_distance, 3)
            self.logger.info(f"Centroid distance: {centroid_distance}")

        if Metric.DISCOURSE_COHERENCE in self.metrics:
            # Discourse Coherence
            discourse_coherence_ = discourse_coherence(hypotheses)
            results["discourse_coherence"] = round(discourse_coherence_, 3)
            self.logger.info(f"Discourse coherence: {discourse_coherence_}")

        if Metric.INTER_SENTENCE_SIMILARITY in self.metrics:
            # Inter-sentence similarity
            inter_sentence_similarity_ = inter_sentence_similarity(
                hypotheses, model=self.model
            )
            results["inter_sentence_similarity"] = round(inter_sentence_similarity_, 3)
            self.logger.info(f"Inter-sentence similarity: {inter_sentence_similarity_}")

        return dict(results)

    def compute_comparison_metrics(self, references, hypotheses) -> dict:
        """Compute metrics for a list of references and hypotheses."""
        results = defaultdict(float)

        if Metric.BLEU in self.metrics:
            # Calculate BLEU score
            bleu_score_ = bleu_score(hypotheses, references)
            results["bleu_score"] = round(bleu_score_, 3)
            self.logger.info(f"BLEU score: {bleu_score_}")

        if Metric.MEAN_LEVENSHTEIN_DISTANCE in self.metrics:
            # Levenshtein distance
            levenshtein_dist = mean_levenshtein_distance(references, hypotheses)
            results["levenshtein_distance"] = round(levenshtein_dist, 3)
            self.logger.info(f"Levenshtein distance: {levenshtein_dist}")

        if Metric.POS_TAG_N_GRAMS_DIVERSITY in self.metrics:
            # POS tag n-grams diversity
            pos_tag_n_grams = pos_tag_n_grams_diversity(references, hypotheses, 2)
            results["pos_tag_n_grams_diversity"] = round(pos_tag_n_grams, 3)
            self.logger.info(f"POS tag n-grams diversity: {pos_tag_n_grams}")

        return dict(results)

    def __apply_framework(
        self, references: list | str = None, hypotheses: list | str = None
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
        self.logger.info("Results: {}".format(results))
        return {"results": results}

    def apply_framework(self, data: list[dict]):
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        results = []
        for item in data:
            reference = item["reference"]
            hypothesis = item["hypothesis"]
            result = self.__apply_framework(reference, hypothesis)
            results.append(result)
        return results

    def apply_framework_to_datasets(
        self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame = None
    ) -> list[dict]:
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        intents: list[str] = [
            intent for intent in dataset_a.intent.unique() if intent is not None
        ]
        self.logger.info(intents)
        results: list[dict] = []
        self.logger.info("Evaluating intents: {}".format(intents))
        for intent in tqdm(intents):
            self.logger.info("Evaluating intent: {}".format(intent))

            references = None
            hypotheses = []

            hypotheses = dataset_a[dataset_a.intent == intent].text.tolist()
            self.logger.info("Number of hypotheses: {}".format(len(hypotheses)))

            if dataset_b:
                references = dataset_b[dataset_b.intent == intent].text.tolist()

            if not hypotheses:
                self.logger.warning(
                    "No data found for intent {}. Skipping evaluation.".format(intent)
                )
                continue
            result = self.__apply_framework(references, hypotheses)
            results.append({intent: result})

        return results


if __name__ == "__main__":

    framework = Framework()
    results = framework.apply_framework(EXAMPLES)
    print(results)
