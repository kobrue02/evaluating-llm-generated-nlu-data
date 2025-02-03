"""
Functions for calculating various metrics for evaluating text generation models.

Author: Konrad Brüggemann
Date: 13.01.2025
"""

import logging
import math
import nltk
import numpy as np
import torch
from collections import defaultdict
from enum import Enum
from nltk import ngrams
from nltk.stem import Cistem
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


cistem = Cistem()
logger = logging.getLogger(__name__)


class Metric(Enum):
    DISTINCT_1 = "distinct_1"
    DISTINCT_2 = "distinct_2"
    TTR = "ttr"
    MOVING_AVERAGE_TTR = "moving_average_ttr"
    BLEU = "bleu"
    INTER_SENTENCE_SIMILARITY = "inter_sentence_similarity"
    DISCOURSE_COHERENCE = "discourse_coherence"
    DISTANCE_TO_CENTROID = "distance_to_centroid"
    SIMILARITY_BY_CLUSTERING = "similarity_by_clustering"
    MEAN_LEVENSHTEIN_DISTANCE = "mean_levenshtein_distance"
    POS_TAG_N_GRAMS_DIVERSITY = "pos_tag_n_grams_diversity"
    AVERAGE_N_OF_TOKENS = "average_n_of_tokens"
    AVERAGE_N_OF_CHARACTERS = "average_n_of_characters"


def _validate_text_input(text: str | list) -> list:
    """Validate and preprocess text input."""
    if isinstance(text, str):
        text = [text]
    return text


def _calculate_ngrams(text: str, n: int) -> list:
    """Calculate n-grams for a given text."""
    tokens = text.split()
    return list(ngrams(tokens, n))


def distinct_n(text: str | list, n: int) -> float:
    """
    Calculate the distinct-n metric of a given text.

    Args:
        text (str | list): The text to calculate the distinct-n of.
        n (int): The n-gram order.

    Returns:
        float: The distinct-n of the text.
    """
    text = _validate_text_input(text)
    text = " ".join(text)
    ngrams_list = _calculate_ngrams(text, n)
    try:
        distinct = len(set(ngrams_list)) / len(ngrams_list)
    except ZeroDivisionError:
        distinct = 0.0
    return distinct


def type_token_ratio(text: str | list) -> float:
    """
    Calculate the type-token ratio of a given text using the Cistem stemmer.

    Args:
        text (str | list): The text to calculate the type-token ratio of.

    Returns:
        float: The type-token ratio of the text.
    """
    text = _validate_text_input(text)
    if not text:
        return 0.0
    tokens = " ".join(text).split()
    types = set([cistem.stem(token) for token in tokens])
    return len(types) / len(tokens)


def moving_average_ttr(text: str | list, window_size: int = 10) -> float:
    """
    Calculate the moving average type-token ratio of a given text using the Cistem stemmer.

    Args:
        text (str | list): The text to calculate the moving average type-token ratio of.
        window_size (int): The window size for the moving average.

    Returns:
        float: The moving average type-token ratio of the text.
    """
    text = _validate_text_input(text)
    tokens = " ".join(text).split()
    ttrs = []
    for i in range(len(tokens) - window_size):
        window = tokens[i : i + window_size]
        types = set([cistem.stem(token) for token in window])
        ttr = len(types) / len(window)
        ttrs.append(ttr)
    return np.mean(ttrs)


def average_n_of_tokens(text: str | list) -> float:
    """
    Calculate the average number of tokens in a list of texts.

    Args:
        text (str | list): The text or list of texts.

    Returns:
        float: The average number of tokens.
    """
    text = _validate_text_input(text)
    token_counts = [len(t.split()) for t in text]
    return np.mean(token_counts)


def average_n_of_characters(text: str | list) -> float:
    """
    Calculate the average number of characters in a list of texts.

    Args:
        text (str | list): The text or list of texts.

    Returns:
        float: The average number of characters.
    """
    text = _validate_text_input(text)
    character_counts = [len(t) for t in text]
    return np.mean(character_counts)


def bleu_score(hypothesis: str | list, reference: str | list) -> float:
    """
    Calculate the BLEU score of a given hypothesis with respect to a reference.

    Args:
        hypothesis (str | list): The hypothesis text.
        reference (str | list): The reference text.

    Returns:
        float: The BLEU score of the hypothesis.
    """
    hypothesis = _validate_text_input(hypothesis)
    reference = _validate_text_input(reference)
    references = [r.split() for r in reference]
    bleu_scores = []
    for text in hypothesis:
        bleu = sentence_bleu(
            references,
            text.split(),
            smoothing_function=SmoothingFunction().method4,
        )
        if math.isnan(bleu):
            bleu = 0.0
        bleu_scores.append(bleu)
    return np.mean(bleu_scores)


def inter_sentence_similarity(sentences: list[str], model=None) -> float:
    """
    Calculate the inter-sentence similarity of a list of sentences using a sentence embedding model.

    Args:
        sentences (list): The list of sentences.
        model (sentence_transformers.SentenceTransformer): The sentence embedding model.

    Returns:
        float: The inter-sentence similarity of the sentences.
    """
    sentences = _validate_text_input(sentences)
    if not sentences or len(sentences) < 2:
        logger.warning(
            "At least two sentences are required for inter-sentence similarity."
        )
        return 0.0

    embeddings = _calculate_embeddings(sentences, model)
    similarities = cosine_similarity(embeddings)

    n = similarities.shape[0]
    sum_of_similarities = np.sum(similarities) - np.sum(np.diag(similarities))
    num_comparisons = n * (n - 1)

    inter_sentence_similarity = sum_of_similarities / num_comparisons
    return (
        inter_sentence_similarity if not math.isnan(inter_sentence_similarity) else 0.0
    )


def _calculate_entity_grid(sentences: list[str]) -> defaultdict:
    """Create an entity grid from a list of sentences."""
    entities = defaultdict(lambda: ["_"] * len(sentences))
    for i, sentence in enumerate(sentences):
        for entity, role in extract_entities(sentence):
            entities[entity][i] = role
    return entities


def extract_entities(sentence: str) -> list[tuple]:
    """Extract entities and their syntactic roles from a sentence."""
    entities = []
    words = nltk.word_tokenize(sentence)
    tagger = nltk.pos_tag(words)  # List of (word, POS) tuples

    for word, pos in tagger:  # Iterate over (word, POS) pairs
        if "NN" in pos:  # Check if POS tag contains 'NN' (noun)
            entities.append((word, "S"))

    return entities


def _calculate_transitions(grid: defaultdict) -> defaultdict:
    """Compute transitions between entity mentions in a grid of sentences."""
    transitions = defaultdict(int)
    for entity_mentions in grid.values():
        for i in range(len(entity_mentions) - 1):
            transition = (entity_mentions[i], entity_mentions[i + 1])
            transitions[transition] += 1
    return transitions


def discourse_coherence(sentences: list[str]) -> float:
    """Calculate the coherence of a list of sentences using the entity grid model."""
    grid = _calculate_entity_grid(sentences)
    transitions = _calculate_transitions(grid)

    total_transitions = sum(transitions.values())
    probabilities = {t: count / total_transitions for t, count in transitions.items()}

    coherence_score = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return coherence_score


def _calculate_embeddings(text: list[str], model) -> np.ndarray:
    """Encode text into embeddings using a sentence transformer model."""
    return model.encode(text)


def distance_to_centroid(hypotheses: list[str], model) -> float:
    """Calculate the distance of hypotheses to the centroid of their embeddings."""
    embeddings = _calculate_embeddings(hypotheses, model)
    centroid = np.mean(embeddings, axis=0)
    distances = [np.linalg.norm(e - centroid) for e in embeddings]
    return np.mean(distances)


def _calculate_clusters(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Cluster embeddings using K-means."""
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def similarity_by_clustering(
    references: list[str], hypotheses: list[str], model
) -> float:
    """Calculate the similarity of hypotheses to reference clusters."""
    reference_embeddings = _calculate_embeddings(references, model)
    reference_centroids = _calculate_clusters(reference_embeddings)

    hypothesis_embeddings = _calculate_embeddings(hypotheses, model)
    similarities = [
        max(cosine_similarity([h], reference_centroids)[0])
        for h in hypothesis_embeddings
    ]
    return np.mean(similarities)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
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


def mean_levenshtein_distance(references: list[str], hypotheses: list[str]) -> float:
    """Calculate the mean Levenshtein distance between hypotheses and references."""
    distances = [levenshtein_distance(r, h) for r in references for h in hypotheses]
    return np.mean(distances)


def pos_tag_n_grams_diversity(hypotheses: list[str]) -> float:
    """Calculate the diversity of POS tag n-grams in a list of hypotheses."""
    scores = []
    if not hypotheses:
        return 0.0
    for hypothesis in hypotheses:
        n_grams = n_grams_of_pos_tags(hypothesis, 2)
        diversity = len(set(n_grams)) / len(n_grams)
        scores.append(diversity)
    return np.mean(scores)


def n_grams_of_pos_tags(text: str, n: int) -> list:
    """Calculate n-grams of POS tags for a given text."""
    pos_tags = get_pos_tags(text)
    return list(ngrams(pos_tags, n))


def get_pos_tags(text: str) -> list:
    """Get POS tags for a given text."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return [tag for token, tag in pos_tags]
