import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Union


class NLUModel:
    def __init__(self, model_path: str, intent_threshold: float = 0.7):
        """
        Initialize the NLU model.

        Args:
            model_path (str): Path to the fine-tuned model or name of the model on Hugging Face's model hub.
            intent_threshold (float): Threshold for intent confidence.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.intent_threshold = intent_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(
        self, text: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict intents for given text(s).

        Args:
            text (Union[str, List[str]]): Input text or list of texts.

        Returns:
            Union[Dict[str, float], List[Dict[str, float]]]: Intent probabilities for each input.
        """
        # Tokenize the input text(s)
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Create a dictionary of intent probabilities for each input
        results = []
        for prob in probs:
            intent_probs = {self.id2label[i]: p.item() for i, p in enumerate(prob)}
            results.append(intent_probs)

        return results[0] if isinstance(text, str) else results

    def is_consistent(self, sample: Dict[str, str]) -> bool:
        """
        Check if the model's prediction is consistent with the intended intent for a given sample.

        Args:
            sample (Dict[str, str]): A dictionary containing 'text' and 'intent' keys.

        Returns:
            bool: True if the prediction is consistent, False otherwise.
        """
        text = sample["text"]
        intended_intent = sample["intent"]

        # Get model predictions
        predictions = self.predict(text)

        # Check if the intended intent has the highest probability
        predicted_intent = max(predictions, key=predictions.get)

        # Check if the probability of the intended intent is above the threshold
        is_above_threshold = predictions[intended_intent] >= self.intent_threshold

        return predicted_intent == intended_intent and is_above_threshold


def nlu_consistency_filtering(
    generated_samples: List[Dict[str, str]], nlu_model: NLUModel
) -> List[Dict[str, str]]:
    """
    Filter generated samples based on NLU model consistency.

    Args:
        generated_samples (List[Dict[str, str]]): List of generated samples, each with 'text' and 'intent' keys.
        nlu_model (NLUModel): The NLU model to use for consistency checking.

    Returns:
        List[Dict[str, str]]: Filtered list of consistent samples.
    """
    filtered_samples = []
    for sample in generated_samples:
        if nlu_model.is_consistent(sample):
            filtered_samples.append(sample)
    return filtered_samples
