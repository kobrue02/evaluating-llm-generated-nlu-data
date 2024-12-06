from framework import calculate_perplexity, calculate_coherence, task_specific_performance, distinct_n
from typing import Dict, Tuple, Union, List, LiteralString


def evaluate_synthetic_data(
        synthetic_data: Union[LiteralString, List[LiteralString], Dict], 
        real_data: Union[LiteralString, List[LiteralString], Dict]
        ) -> Dict:
    """
    Evaluate the synthetic data using the provided metrics.
    Args:
        synthetic_data (str, dict list of str): The synthetic data to evaluate.
        real_data (str, dict or list of str): The real data to compare against.
    Returns:
        dict: The evaluation results.
    """
    results = {}

    if isinstance(synthetic_data, list):
        synthetic_data = ' '.join(synthetic_data)
    if isinstance(real_data, list):
        real_data = ' '.join(real_data)

    if isinstance(synthetic_data, dict):
        if not "intent" in synthetic_data:
            raise ValueError("The synthetic data dictionary must have an 'intent' key.")
        for intent in synthetic_data["intent"]:
            synthetic_text = synthetic_data["intent"][intent]
            real_text = real_data["intent"][intent]
            results[intent] = evaluate_synthetic_data(synthetic_text, real_text)
        return results

    perplexity, diversity, coherence = calculate_metrics(synthetic_data)
    results["perplexity"] = perplexity
    results["diversity"] = diversity
    results["coherence"] = coherence
    return results

def calculate_metrics(text: LiteralString) -> Tuple:
    perplexity = calculate_perplexity(text=text)
    diversity = distinct_n(text=text, n=3)
    coherence = calculate_coherence(text=text)
    return perplexity, diversity, coherence


if __name__ == "__main__":
    # Example usage
    synthetic_text = [
        "Turn on the AC in the back of the car.",
        "Play some music.",
        "Navigate to the nearest gas station.",
        "What's the weather like today?"
        ]
    real_text = ["Your real text here."]

    results = evaluate_synthetic_data(synthetic_text, real_text)
    print(results)