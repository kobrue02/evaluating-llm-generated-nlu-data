from collections import OrderedDict
from transformers import pipeline
from tqdm import tqdm
from typing import List

from bin.data_generation.construct_prompt import Prompt, load_prompt
from bin.utils.types import DataSet
from bin.utils.exceptions import MalformedOutputError

import ast
import logging
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
        self.logger = logging.getLogger(__name__)

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
            output_queries = self._parse_output(output[0]["generated_text"])
        except (ValueError, SyntaxError, TypeError) as e:
            raise MalformedOutputError(f"Error parsing generated queries: {e}")

        synthetic_data.extend(
            output_queries, labels=[prompt.intent] * len(output_queries)
        )
        return synthetic_data
    
    def _parse_output(self, output_text: str) -> List[str]:
        """
        Parse the output text and extract the queries.

        Args:
            output_text (str): The generated text from the model.

        Returns:
            List[str]: A list of parsed queries.
        """
        try:
            output_queries = ast.literal_eval(output_text)
        except (ValueError, SyntaxError):
            # If literal_eval fails, try to extract queries using string manipulation
            if "Here are the queries:" in output_text:
                output_queries = output_text.split("Here are the queries:")[1].strip().split("\n")
            else:
                output_queries = output_text.strip().split("\n")

        if isinstance(output_queries, str):
            output_queries = [output_queries]
        elif isinstance(output_queries, list):
            output_queries = [query for query in output_queries if query and isinstance(query, str)]
        else:
            self.logger.error(f"Unexpected output format: {output_queries}")
            raise ValueError("Unexpected output format")

        return output_queries

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
                    self.logger.info(f"Generated {len(batch_data)} samples for {intent}")
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
