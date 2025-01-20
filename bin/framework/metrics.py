"""
Functions for calculating various metrics for evaluating text generation models.

Author: Konrad Brüggemann
Date: 13.01.2025
"""

import math
import nltk
import numpy as np
from sklearn.cluster import KMeans
import torch

from collections import defaultdict
from nltk import ngrams
from nltk.stem import Cistem
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


cistem = Cistem()


def calculate_perplexity(
    text: str | list, model=None, tokenizer=None, base=2, max_perplexity=10000
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
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
        scores.append(1 - (math.log(perplexity, base) / math.log(max_perplexity, base)))
    return np.mean(scores)


def distinct_n(text: str | list, n):
    """
    Calculate the distinct-n metric of a given text.
    Args:
        text (str): The text to calculate the distinct-n of.
        n (int): The n-gram order.
    Returns:
        float: The distinct-n of the text.
    """
    if isinstance(text, list):
        text = " ".join(text)
    tokens = text.split()
    ngrams_list = list(ngrams(tokens, n))
    try:
        distinct = len(set(ngrams_list)) / len(ngrams_list)
    except ZeroDivisionError:
        distinct = 0.0
    return distinct


def type_token_ratio(text):
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
        types = set([cistem.stem(token) for token in tokens])
        ttr = len(types) / len(tokens)
        ttrs.append(ttr)
    return np.mean(ttrs)


def moving_average_ttr(text, window_size=100):
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
            window_types = set([cistem.stem(token) for token in window])
            ttr = len(window_types) / len(window)
            if not math.isnan(ttr):
                ttrs.append(ttr)
    if not ttrs:
        return 0.0

    return np.mean(ttrs)


def average_n_of_tokens(text: str | list) -> float:
    """
    Calculate the average number of tokens in a list of texts.
    Args:
        text (str | list): The text or list of texts.
        n (int): The number of tokens to calculate the average of.
    Returns:
        float: The average number of tokens.
    """
    if isinstance(text, str):
        text = [text]
    token_counts = [len(t.split()) for t in text]
    return np.mean(token_counts)


def average_n_of_characters(text: str | list) -> float:
    """
    Calculate the average number of characters in a list of texts.
    Args:
        text (str | list): The text or list of texts.
        n (int): The number of characters to calculate the average of.
    Returns:
        float: The average number of characters.
    """
    if isinstance(text, str):
        text = [text]
    character_counts = [len(t) for t in text]
    return np.mean(character_counts)


def bleu_score(hypothesis: str, reference: str) -> float:
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


def task_specific_performance(train_data, test_data, model=None):
    raise NotImplementedError


def inter_sentence_similarity(sentences: list[str], model=None):
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
    embeddings = model.encode(sentences)

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


def create_entity_grid(sentences):
    """Create an entity grid from a list of sentences."""
    entities = defaultdict(lambda: ["_"] * len(sentences))

    for i, sentence in enumerate(sentences):
        for entity, role in extract_entities(sentence):
            entities[entity][i] = role

    return entities


def extract_entities(sentence):
    """Extract entities and their syntactic roles from a sentence."""
    # This is a simplified version. In practice, you'd use NLP tools
    # to extract entities and their syntactic roles (S, O, X)
    entities = []
    for word in sentence.split():
        if word.istitle():
            entities.append((word, "S"))  # Assume all capitalized words are subjects
    return entities


def compute_transitions(grid):
    """Compute transitions between entity mentions in a grid of sentences."""
    transitions = defaultdict(int)
    for entity_mentions in grid.values():
        for i in range(len(entity_mentions) - 1):
            transition = (entity_mentions[i], entity_mentions[i + 1])
            transitions[transition] += 1
    return transitions


def discourse_coherence(sentences):
    """Calculate the coherence of a list of sentences using the entity grid model."""
    grid = create_entity_grid(sentences)
    transitions = compute_transitions(grid)

    # Calculate probabilities of transitions
    total_transitions = sum(transitions.values())
    probabilities = {t: count / total_transitions for t, count in transitions.items()}

    # Calculate coherence score (e.g., using entropy)
    coherence_score = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

    return coherence_score


def distance_to_centroid(hypotheses, model):
    """Calculate the distance of hypotheses to the centroid of their embeddings."""
    embeddings = model.encode(hypotheses)
    centroid = np.mean(embeddings, axis=0)
    distances = [np.linalg.norm(e - centroid) for e in embeddings]
    return np.mean(distances)


def similarity_by_clustering(references, hypotheses, model):
    """Calculate the similarity of hypotheses to reference clusters."""
    # Cluster embeddings of references
    reference_embeddings = model.encode(references)
    reference_centroids = cluster_embeddings(reference_embeddings)

    # Calculate similarity of hypotheses to reference clusters
    hypothesis_embeddings = model.encode(hypotheses)
    similarities = []
    for h in hypothesis_embeddings:
        similarities.append(max(cosine_similarity(h, c) for c in reference_centroids))

    return np.mean(similarities)


def cluster_embeddings(embeddings):
    """Cluster embeddings using K-means."""
    # Cluster embeddings using K-means
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def levenshtein_distance(s1, s2):
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


def cartesian_product(list_a: list[str], list_b: list[str]):
    """Calculate the Cartesian product of two lists."""
    return [(a, b) for a in list_a for b in list_b]


def mean_levenshtein_distance(references, hypotheses):
    """Calculate the mean Levenshtein distance between hypotheses and references."""
    c_product = cartesian_product(references, hypotheses)
    distances = [levenshtein_distance(r, h) for r, h in c_product]
    return np.mean(distances)


def pos_tag_n_grams_diversity(references, hypotheses, n):
    """Calculate the diversity of n-grams of POS tags in hypotheses with respect to references."""
    for sample in references:
        reference_n_grams = n_grams_of_pos_tags(sample, n)
    for sample in hypotheses:
        hypothesis_n_grams = n_grams_of_pos_tags(sample, n)
    reference_n_grams = set(reference_n_grams)
    hypothesis_n_grams = set(hypothesis_n_grams)
    try:
        diversity = len(hypothesis_n_grams.difference(reference_n_grams)) / len(
            hypothesis_n_grams
        )
    except ZeroDivisionError:
        diversity = 0.0
    return diversity


def n_grams_of_pos_tags(text, n):
    """Calculate n-grams of POS tags for a given text."""
    pos_tags = get_pos_tags(text)
    n_grams = list(ngrams(pos_tags, n))
    return n_grams


def get_pos_tags(text):
    """Get POS tags for a given text."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return [tag for token, tag in pos_tags]
