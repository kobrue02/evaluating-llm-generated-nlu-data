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
import pandas as pd

torch.random.manual_seed(0)


class DataGenerationModel:
    def __init__(self, *args, model=None, tokenizer=None, reference_dataset: pd.DataFrame = None):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_dataset = reference_dataset

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.logger.info("Initializing pipeline")
        try:
            self.pipe = pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer
            )
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
            raise

    def generate_synthetic_data(self, prompt: Prompt) -> DataSet:
        synthetic_data = DataSet()

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        messages = list(prompt)
        self.logger.info("Generating text")
        try:
            output = self.pipe(messages, **generation_args)
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            raise

        self.logger.info("Parsing output")
        try:
            output_queries = self._parse_output(output[0]["generated_text"])
        except (ValueError, SyntaxError, TypeError) as e:
            self.logger.error(f"Error parsing generated queries: {e}")
            raise MalformedOutputError(f"Error parsing generated queries: {e}")

        # Ensure queries and labels match in length
        if len(output_queries) != len([prompt.intent] * len(output_queries)):
            self.logger.error("Mismatch between generated queries and intent labels")
            raise ValueError("Mismatch between generated queries and intent labels")

        synthetic_data.extend(
            output_queries, labels=[prompt.intent] * len(output_queries)
        )
        return synthetic_data

    def build_dataset_from_intents(
        self, prompt_id: str, intents: list[str], samples_per_intent: int = 10
    ) -> DataSet:
        synthetic_data = DataSet()
        max_retries = 10

        for intent in tqdm(intents, desc="Processing intents"):
            self.logger.info(f"Generating data for intent: {intent}")
            unique_samples = OrderedDict()
            remaining_samples = samples_per_intent
            generated_queries = []
            retries = 0

            while remaining_samples > 0 and retries < max_retries:
                retries += 1
                batch_size = min(10, remaining_samples)
                prompt = load_prompt(
                    id=prompt_id,
                    intent=intent,
                    num_samples=batch_size,
                    generated_queries=generated_queries,
                )
                try:
                    batch_data = self.generate_synthetic_data(prompt)
                    self.logger.info(
                        f"Generated {len(batch_data)} samples for {intent}"
                    )
                    self.logger.info(f"The generated samples are: {batch_data}")
                    for sample in batch_data:
                        if sample not in unique_samples:
                            unique_samples[sample] = None
                            generated_queries.append(sample)
                            remaining_samples -= 1
                        if remaining_samples == 0:
                            break
                    self.logger.info(
                        f"Number of queries: {len(generated_queries)}, Number of labels: {len([prompt.intent] * len(generated_queries))}"
                    )
                except MalformedOutputError as e:
                    self.logger.warning(f"MalformedOutputError for {intent}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for {intent}: {e}")
                    continue
                self.logger.info(
                    f"Remaining samples for intent {intent}: {remaining_samples}"
                )
            synthetic_data += DataSet(
                data=generated_queries, labels=[prompt.intent] * len(generated_queries)
            )

        # Final validation to ensure consistent dataset
        if len(synthetic_data.data) != len(synthetic_data.labels):
            self.logger.error("Mismatch between queries and labels in dataset.")
            raise ValueError("Mismatch between queries and labels in dataset.")

        return synthetic_data

    def build_dataset_from_reference(self, prompt_id: str, samples_per_intent: int = 10) -> DataSet:

        synthetic_data = DataSet()
        max_retries = 10

        intents = self.reference_dataset.intent.unique()

        for intent in tqdm(intents, desc="Processing intents"):
            self.logger.info(f"Generating data for intent: {intent}")
            unique_samples = OrderedDict()
            remaining_samples = samples_per_intent
            generated_queries = []
            retries = 0

            examples = self.reference_dataset[self.reference_dataset.intent == intent].text.tolist()
            if prompt_id == "few_shot_simple":
                examples = examples[:3]
            elif prompt_id == "one_shot_simple":
                examples = examples[:1]

            while remaining_samples > 0 and retries < max_retries:
                retries += 1
                batch_size = min(10, remaining_samples)
                prompt = load_prompt(
                    id=prompt_id,
                    intent=intent,
                    num_samples=batch_size,
                    generated_queries=generated_queries,
                    examples=examples
                )
                try:
                    batch_data = self.generate_synthetic_data(prompt)
                    self.logger.info(
                        f"Generated {len(batch_data)} samples for {intent}"
                    )
                    self.logger.info(f"The generated samples are: {batch_data}")
                    for sample in batch_data:
                        if sample not in unique_samples:
                            unique_samples[sample] = None
                            generated_queries.append(sample)
                            remaining_samples -= 1
                        if remaining_samples == 0:
                            break
                    self.logger.info(
                        f"Number of queries: {len(generated_queries)}, Number of labels: {len([prompt.intent] * len(generated_queries))}"
                    )
                except MalformedOutputError as e:
                    self.logger.warning(f"MalformedOutputError for {intent}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for {intent}: {e}")
                    continue
                self.logger.info(
                    f"Remaining samples for intent {intent}: {remaining_samples}"
                )
            synthetic_data += DataSet(
                data=generated_queries, labels=[prompt.intent] * len(generated_queries)
            )

        # Final validation to ensure consistent dataset
        if len(synthetic_data.data) != len(synthetic_data.labels):
            self.logger.error("Mismatch between queries and labels in dataset.")
            raise ValueError("Mismatch between queries and labels in dataset.")

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
            self.logger.warning("Fallback to line splitting for output parsing.")
            output_queries = output_text.strip().split("\n")

        if isinstance(output_queries, str):
            output_queries = [output_queries]

        elif not isinstance(output_queries, list):
            self.logger.error(f"Unexpected output format: {type(output_queries)}")
            raise ValueError("Unexpected output format")

        # Ensure only valid strings are returned
        return [
            query.strip()
            for query in output_queries
            if isinstance(query, str) and query.strip()
        ]
