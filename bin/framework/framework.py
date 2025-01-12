import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import logging
import math
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from nltk import ngrams
from nltk.stem import Cistem
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm

from bin.utils.types import DataSet
from bin.framework.example_data import EXAMPLES

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

    def __init__(self):
        self.cistem = Cistem()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.g_model = AutoModelForMaskedLM.from_pretrained(
            "google-bert/bert-base-german-cased"
        )
        self.g_tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-german-cased"
        )
        self.logger = logging.getLogger(__name__)
        logging.getLogger(__name__).setLevel(logging.INFO)

    def calculate_perplexity(
        self, text: str | list, model=None, tokenizer=None, base=2, max_perplexity=10000
    ):
        """
        Calculate the perplexity of a given text using BERT's masked language modeling.

        Args:
            text (str): The text to calculate the perplexity of.
            model (transformers.PreTrainedModel): The BERT model. By default, uses bert-base-uncased.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer. By default, uses bert-base-uncased.

        Returns:
            float: The perplexity of the text.
        """

        if isinstance(text, str):
            text = [text]
        scores = []
        for t in text:
            if not t:
                return 0.0
            inputs = self.g_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                outputs = self.g_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = math.exp(loss.item())
            scores.append(
                1 - (math.log(perplexity, base) / math.log(max_perplexity, base))
            )
        return np.mean(scores)

    def distinct_n(self, text: str | list, n):
        """
        Calculate the distinct-n metric of a given text.
        Args:
            text (str): The text to calculate the distinct-n of.
            n (int): The n-gram order.
        Returns:
            float: The distinct-n of the text.
        """
        if isinstance(text, str):
            text = [text]
        distincts = []
        for t in text:
            tokens = t.split()
            ngrams_list = list(ngrams(tokens, n))
            try:
                distinct = len(set(ngrams_list)) / len(ngrams_list)
            except ZeroDivisionError:
                distinct = 0.0
            distincts.append(distinct)
        return np.mean(distincts)

    def type_token_ratio(self, text):
        """
        Calculate the type-token ratio of a given text using the Cistem stemmer.
        Args:
            text (str): The text to calculate the type-token ratio of.
        Returns:
            float: The type-token ratio of the text.
        """
        if not text:
            return 0.0
        elif isinstance(text, str):
            text = [text]
        ttrs = []
        for t in text:
            tokens = t.split()
            types = set([self.cistem.stem(token) for token in tokens])
            ttr = len(types) / len(tokens)
            ttrs.append(ttr)
        return np.mean(ttrs)

    def moving_average_ttr(self, text, window_size=100):
        """
        Calculate the moving average type-token ratio of a given text using the Cistem stemmer.
        Args:
            text (str): The text to calculate the moving average type-token ratio of.
            window_size (int): The window size for the moving average.
        Returns:
            list: The moving average type-token ratio of the text.
        """
        ttrs = []
        if isinstance(text, str):
            text = [text]

        for t in text:
            tokens = t.split()
            for i in range(len(tokens) - window_size):
                window = tokens[i : i + window_size]
                window_types = set([self.cistem.stem(token) for token in window])
                ttr = len(window_types) / len(window)
                if not math.isnan(ttr):
                    ttrs.append(ttr)
        if not ttrs:
            return 0.0

        return np.mean(ttrs)

    def bleu_score(self, hypothesis: str, reference: str) -> float:
        """
        Calculate the BLEU score of a given hypothesis with respect to a reference.
        Args:
            hypothesis (str): The hypothesis text.
            reference (str): The reference text.
        Returns:
            float: The BLEU score of the hypothesis.
        """

        if isinstance(hypothesis, str):
            hypotheses = [hypothesis]
        else:
            hypotheses = hypothesis

        if isinstance(reference, str):
            references = [reference.split()]
        else:
            references = [r.split() for r in reference]

        bleu_scores = []
        for text in hypotheses:
            bleu = sentence_bleu(
                references,
                text.split(),
                smoothing_function=SmoothingFunction().method4,
            )
            if math.isnan(bleu):
                bleu = 0.0
            bleu_scores.append(bleu)
        return np.mean(bleu_scores)

    def task_specific_performance(
        self, train_data: DataSet, test_data: DataSet, model=None
    ):
        """
        Calculate the performance of a model on a task-specific dataset.
        Args:
            train_data (DataSet): The training dataset.
            test_data (DataSet): The test dataset.
            model (Model): The model to evaluate.
        Returns:
            float: The performance of the model on the test dataset.
        """
        # Train model on synthetic data
        model.train(train_data)
        # Evaluate on real test data
        results = model.evaluate(test_data)
        # Return relevant metrics (e.g., F1 score)
        metrics = results["metrics"]
        return metrics["f1"]

    def inter_sentence_similarity(self, sentences: list[str], model=None):
        """
        Calculate the inter-sentence similarity of a list of sentences using a sentence embedding model.
        Args:
            sentences (list): The list of sentences.
            model (sentence_transformers.SentenceTransformer): The sentence embedding model.
        Returns:
            float: The inter-sentence similarity of the sentences.
        """

        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences or len(sentences) < 2:
            raise ValueError(
                "At least two sentences are required to compute inter-sentence similarity. Provided sentences: {}".format(
                    str(sentences)
                )
            )

        # Generate embeddings
        embeddings = self.model.encode(sentences)

        # Ensure embeddings are 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Calculate cosine similarities
        similarities = cosine_similarity(embeddings)

        # Exclude diagonal (self-similarity) elements and compute the mean
        n = similarities.shape[0]
        sum_of_similarities = np.sum(similarities) - np.sum(
            np.diag(similarities)
        )  # Exclude diagonal
        num_comparisons = n * (n - 1)  # Total number of off-diagonal elements

        inter_sentence_similarity = sum_of_similarities / num_comparisons
        if math.isnan(inter_sentence_similarity):
            return 0.0
        return inter_sentence_similarity

    def create_entity_grid(self, sentences):
        """Create an entity grid from a list of sentences."""
        entities = defaultdict(lambda: ["_"] * len(sentences))

        for i, sentence in enumerate(sentences):
            for entity, role in self.extract_entities(sentence):
                entities[entity][i] = role

        return entities

    def extract_entities(self, sentence):
        """Extract entities and their syntactic roles from a sentence."""
        # This is a simplified version. In practice, you'd use NLP tools
        # to extract entities and their syntactic roles (S, O, X)
        entities = []
        for word in sentence.split():
            if word.istitle():
                entities.append(
                    (word, "S")
                )  # Assume all capitalized words are subjects
        return entities

    def compute_transitions(self, grid):
        """Compute transitions between entity mentions in a grid of sentences."""
        transitions = defaultdict(int)
        for entity_mentions in grid.values():
            for i in range(len(entity_mentions) - 1):
                transition = (entity_mentions[i], entity_mentions[i + 1])
                transitions[transition] += 1
        return transitions

    def discourse_coherence(self, sentences):
        """Calculate the coherence of a list of sentences using the entity grid model."""
        grid = self.create_entity_grid(sentences)
        transitions = self.compute_transitions(grid)

        # Calculate probabilities of transitions
        total_transitions = sum(transitions.values())
        probabilities = {
            t: count / total_transitions for t, count in transitions.items()
        }

        # Calculate coherence score (e.g., using entropy)
        coherence_score = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

        return coherence_score

    def distance_to_centroid(self, hypotheses):
        """Calculate the distance of hypotheses to the centroid of their embeddings."""
        embeddings = self.model.encode(hypotheses)
        centroid = np.mean(embeddings, axis=0)
        distances = [np.linalg.norm(e - centroid) for e in embeddings]
        return np.mean(distances)

    def similarity_by_clustering(self, references, hypotheses):
        """Calculate the similarity of hypotheses to reference clusters."""
        # Cluster embeddings of references
        reference_embeddings = self.model.encode(references)
        reference_centroids = self.cluster_embeddings(reference_embeddings)

        # Calculate similarity of hypotheses to reference clusters
        hypothesis_embeddings = self.model.encode(hypotheses)
        similarities = []
        for h in hypothesis_embeddings:
            similarities.append(
                max(self.cosine_similarity(h, c) for c in reference_centroids)
            )

        return np.mean(similarities)

    def cluster_embeddings(self, embeddings):
        """Cluster embeddings using K-means."""
        # Cluster embeddings using K-means
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_

    def __apply_framework(self, references: list | str, hypotheses: list | str) -> dict:
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        results = defaultdict(float)
        # Calculate perplexity
        perplexity = self.calculate_perplexity(hypotheses)
        results["perplexity"] = round(perplexity, 3)
        self.logger.info(f"Perplexity: {perplexity}")
        # Calculate distinct-1
        distinct_1 = self.distinct_n(hypotheses, 1)
        results["distinct_1"] = round(distinct_1, 3)
        self.logger.info(f"Distinct-1: {distinct_1}")
        # Calculate distinct-2
        distinct_2 = self.distinct_n(hypotheses, 2)
        results["distinct_2"] = round(distinct_2, 3)
        self.logger.info(f"Distinct-2: {distinct_2}")
        # Calculate type-token ratio
        ttr = self.type_token_ratio(hypotheses)
        results["ttr"] = round(ttr, 3)
        self.logger.info(f"TTR: {ttr}")
        # Calculate moving average TTR
        moving_average_ttr = self.moving_average_ttr(hypotheses)
        results["moving_average_ttr"] = round(moving_average_ttr, 3)
        self.logger.info(f"Moving average TTR: {moving_average_ttr}")
        # Calculate BLEU score
        bleu_score = self.bleu_score(hypotheses, references)
        results["bleu_score"] = round(bleu_score, 3)
        self.logger.info(f"BLEU score: {bleu_score}")
        # Discourse coherence
        discourse_coherence = self.discourse_coherence(hypotheses)
        results["discourse_coherence"] = round(discourse_coherence, 3)
        self.logger.info(f"Discourse coherence: {discourse_coherence}")
        # Inter-sentence similarity
        inter_sentence_similarity = self.inter_sentence_similarity(hypotheses)
        results["inter_sentence_similarity"] = round(inter_sentence_similarity, 3)
        self.logger.info(f"Inter-sentence similarity: {inter_sentence_similarity}")
        # Centroid distance
        centroid_distance = self.distance_to_centroid(hypotheses)
        results["centroid_distance"] = round(centroid_distance, 3)
        self.logger.info(f"Centroid distance: {centroid_distance}")
        # Pos tag n-grams diversity
        pos_tag_n_grams_diversity = self.pos_tag_n_grams_diversity(
            references, hypotheses, 2
        )
        results["pos_tag_n_grams_diversity"] = round(pos_tag_n_grams_diversity, 3)
        self.logger.info(f"POS tag n-grams diversity: {pos_tag_n_grams_diversity}")
        # Levenshtein distance
        levenshtein_distance = self.mean_levenshtein_distance(references, hypotheses)
        results["levenshtein_distance"] = round(levenshtein_distance, 3)
        self.logger.info(f"Levenshtein distance: {levenshtein_distance}")

        joint_score = np.mean(
            [value for value in results.values() if not math.isnan(value)]
        )
        self.logger.info(f"Joint score: {joint_score}")
        return {"results": dict(results), "joint_score": round(joint_score, 3)}

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

    def levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def mean_levenshtein_distance(self, references, hypotheses):
        """Calculate the mean Levenshtein distance between hypotheses and references."""
        distances = [
            self.levenshtein_distance(r, h) for r, h in zip(references, hypotheses)
        ]
        return np.mean(distances)

    def pos_tag_n_grams_diversity(self, references, hypotheses, n):
        """Calculate the diversity of n-grams of POS tags in hypotheses with respect to references."""
        for sample in references:
            reference_n_grams = self.n_grams_of_pos_tags(sample, n)
        for sample in hypotheses:
            hypothesis_n_grams = self.n_grams_of_pos_tags(sample, n)
        reference_n_grams = set(reference_n_grams)
        hypothesis_n_grams = set(hypothesis_n_grams)
        try:
            diversity = len(hypothesis_n_grams.difference(reference_n_grams)) / len(
                hypothesis_n_grams
            )
        except ZeroDivisionError:
            diversity = 0.0
        return diversity

    def n_grams_of_pos_tags(self, text, n):
        """Calculate n-grams of POS tags for a given text."""
        pos_tags = self.get_pos_tags(text)
        n_grams = list(ngrams(pos_tags, n))
        return n_grams

    def get_pos_tags(self, text):
        """Get POS tags for a given text."""
        # This is a simplified version. In practice, you'd use NLP tools
        # to extract POS tags
        return ["NN" for _ in text.split()]

    def apply_framework_to_datasets(
        self, golden_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> list[dict]:
        """
        Apply the framework to a text generation model.
        Returns:
            dict: The results of the evaluation.
        """
        intents: list[str] = [
            intent for intent in golden_data.intent.unique() if intent is not None
        ]
        self.logger.info(intents)
        results: list[dict] = []
        self.logger.info("Evaluating intents: {}".format(intents))
        for intent in tqdm(intents):
            references = golden_data[golden_data.intent == intent].text.tolist()
            hypotheses = generated_data[generated_data.intent == intent].text.tolist()
            if not hypotheses or not references:
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
