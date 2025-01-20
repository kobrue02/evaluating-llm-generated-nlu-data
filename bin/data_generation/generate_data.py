from transformers import pipeline
from tqdm import tqdm
from collections import OrderedDict

from bin.data_generation.construct_prompt import Prompt, load_prompt
from bin.utils.types import DataSet
from bin.utils.exceptions import MalformedOutputError

import ast
import torch

torch.random.manual_seed(0)


class DataGenerationModel:
    """
    A class to generate synthetic data using a language model.
    """

    def __init__(self, *args, model=None, tokenizer=None):
        """
        Initialize the data generation model.
        Args:
            model (transformers.PreTrainedModel): The language model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_synthetic_data(self, prompt: Prompt) -> DataSet:
        """
        Generate synthetic data using the model.

        Args:
            prompt (Prompt): The prompt to generate data from.

        Returns:
            DataSet: The synthetic data.
        """
        synthetic_data = DataSet()

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        messages = list(prompt)
        output = pipe(messages, **generation_args)

        try:
            output_queries = ast.literal_eval(output[0]["generated_text"])
            if "Here are the queries:" in output_queries:
                output_queries = [
                    query
                    for query in output_queries[
                        output_queries.index("Here are the queries:") + 1 :
                    ]
                    if query
                ]
            if any([isinstance(query, list) for query in output_queries]):
                output_queries = [
                    query
                    for query in [
                        sublist
                        for sublist in output_queries
                        if isinstance(sublist, list)
                    ][0]
                ]

            # If the output is a string, convert it to a list
            if isinstance(output_queries, str):
                output_queries = ast.literal_eval(output_queries)
            else:
                output_queries = [query for query in output_queries if query]

        except (ValueError, SyntaxError, TypeError) as e:
            raise MalformedOutputError(f"Error parsing generated queries: {e}")

        synthetic_data.extend(
            output_queries, labels=[prompt.intent] * len(output_queries)
        )
        return synthetic_data

    def build_dataset_from_intents(
        self, prompt_id: str, intents: list[str], samples_per_intent: int = 10
    ) -> DataSet:
        """
        Generate synthetic data from a list of intents.

        Args:
            prompt_id (str): The ID of the prompt to use.
            intents (list[str]): The intents to generate data from.
            samples_per_intent (int): The number of samples to generate per intent.

        Returns:
            DataSet: The synthetic data.
        """
        synthetic_data = DataSet()
        for intent in tqdm(intents):
            unique_samples = OrderedDict()
            remaining_samples = samples_per_intent

            while remaining_samples > 0:
                batch_size = min(10, remaining_samples)
                prompt = load_prompt(
                    id=prompt_id, intent=intent, num_samples=batch_size
                )
                try:
                    batch_data = self.generate_synthetic_data(prompt)
                    for sample in batch_data:
                        if sample not in unique_samples:
                            unique_samples[sample] = None
                            remaining_samples -= 1
                        if remaining_samples == 0:
                            break
                except MalformedOutputError:
                    continue

            synthetic_data += DataSet(list(unique_samples.keys()))

        return synthetic_data
