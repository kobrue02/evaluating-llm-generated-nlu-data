from collections import OrderedDict
import re
from transformers import pipeline
from tqdm import tqdm
from typing import List, Optional

from bin.data_generation.construct_prompt import Prompt, load_prompt
from bin.utils.types import DataSet
from bin.utils.exceptions import MalformedOutputError

import ast
import logging
import torch
import pandas as pd

torch.random.manual_seed(0)


def process_string(text: str) -> str:
    # if the text starts with a number, remove it
    if text[0].isdigit():
        text = text.split(" ", 1)[1]
    # remove punctuation except letters, numbers, spaces, and umlauts
    text = re.sub(r"[^\w\säöüÄÖÜ]", "", text)
    # remove leading/trailing whitespaces
    text = text.strip()
    # lowercase
    return text.lower()

class DataGenerationModel:
    def __init__(
        self,
        *args,
        model=None,
        tokenizer=None,
        reference_dataset: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_dataset = reference_dataset

        self._initialize_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.pipe = self._initialize_pipeline()

    def _initialize_logger(self):
        """Initialize the logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def _initialize_pipeline(self):
        """Initialize the text generation pipeline."""
        self.logger.info("Initializing pipeline")
        try:
            return pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer
            )
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
            raise

    def generate_synthetic_data(self, prompt: Prompt) -> DataSet:
        """Generate synthetic data from a prompt."""
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
        self, prompt_id: str, intents: List[str], samples_per_intent: int = 10
    ) -> DataSet:
        """Build a dataset from a list of intents."""
        synthetic_data = DataSet()

        for intent in tqdm(intents, desc="Processing intents"):
            self.logger.info(f"Generating data for intent: {intent}")
            synthetic_data += self._process_intent(
                prompt_id, intent, samples_per_intent
            )

        self._validate_dataset(synthetic_data)
        return synthetic_data

    def build_dataset_from_reference(
        self, prompt_id: str, samples_per_intent: int = 10
    ) -> DataSet:
        """Build a dataset using a reference dataset."""
        synthetic_data = DataSet()

        if self.reference_dataset is None:
            raise ValueError("Reference dataset is not provided.")

        intents = self.reference_dataset.intent.unique()

        for intent in tqdm(intents, desc="Processing intents"):
            self.logger.info(f"Generating data for intent: {intent}")
            examples = self._get_examples_from_reference(intent, prompt_id)
            synthetic_data += self._process_intent(
                prompt_id, intent, samples_per_intent, examples
            )

        self._validate_dataset(synthetic_data)
        return synthetic_data

    def _process_intent(
        self,
        prompt_id: str,
        intent: str,
        samples_per_intent: int,
        examples: Optional[List[str]] = None,
    ) -> DataSet:
        """Process a single intent to generate synthetic data."""
        unique_samples = OrderedDict()
        remaining_samples = samples_per_intent
        generated_queries = []
        max_retries = 10
        retries = 0

        while remaining_samples > 0 and retries < max_retries:
            retries += 1
            batch_size = min(10, remaining_samples)
            prompt = load_prompt(
                id=prompt_id,
                intent=intent,
                num_samples=batch_size,
                generated_queries=generated_queries,
                examples=examples,
            )
            try:
                batch_data = self._generate_batch_data(prompt, intent)
                for sample in batch_data:
                    if sample not in unique_samples:
                        unique_samples[sample] = None
                        generated_queries.append(sample)
                        remaining_samples -= 1
                    if remaining_samples == 0:
                        break
            except MalformedOutputError as e:
                self.logger.warning(f"MalformedOutputError for {intent}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for {intent}: {e}")
                continue

        return DataSet(data=generated_queries, labels=[intent] * len(generated_queries))

    def _generate_batch_data(self, prompt: Prompt, intent: str) -> List[str]:
        """Generate and validate batch data for a prompt."""
        batch_data = self.generate_synthetic_data(prompt)
        self.logger.info(f"Generated {len(batch_data)} samples for {intent}")
        self.logger.info(f"The generated samples are: {batch_data}")
        return batch_data.data

    def _get_examples_from_reference(self, intent: str, prompt_id: str) -> List[str]:
        """Retrieve examples from the reference dataset."""
        examples = self.reference_dataset[
            self.reference_dataset.intent == intent
        ].text.tolist()
        if prompt_id == "few_shot_simple":
            return examples[:3]
        elif prompt_id == "one_shot_simple":
            return examples[:1]
        return examples

    def _validate_dataset(self, dataset: DataSet):
        """Validate the dataset to ensure consistency."""
        if len(dataset.data) != len(dataset.labels):
            self.logger.error("Mismatch between queries and labels in dataset.")
            raise ValueError("Mismatch between queries and labels in dataset.")

    def _parse_output(self, output_text: str) -> List[str]:
        """
        Parse the output text and extract the queries.

        Args:
            output_text (str): The generated text from the model.

        Returns:
            List[str]: A list of parsed queries.
        """
        try:
            # Handle case where "Here are the queries" is in the text
            if "Here are the queries" in output_text:
                output_text = output_text.split("Here are the queries")[1]

            if "here are the 10 additional queries for the intent" in output_text:
                # find the first \n\n after the phrase
                start = output_text.find("here are the 10 additional queries for the intent")
                end = output_text.find("\n\n", start)
                output_text = output_text[end:]
                output_queries = output_text.split("\n")
                return [process_string(q) for q in output_queries if q.strip()]


            # Extract content between first '[' and last ']'
            start = output_text.find("[")
            end = output_text.rfind("]")
            if start != -1 and end != -1:
                output_text = output_text[start : end + 1]

            # Remove unwanted characters (e.g., backslashes, quotes, brackets)
            output_text = re.sub(
                r"\\'", "'", output_text
            )  # Replace escaped single quotes
            output_text = re.sub(
                r"\\\"", '"', output_text
            )  # Replace escaped double quotes

            # Try to parse as a Python literal
            output_queries = ast.literal_eval(output_text)

            # Ensure output is a list
            if not isinstance(output_queries, list):
                raise ValueError("Unexpected output format")

            # Clean each query
            output_queries = [process_string(q) for q in output_queries if q.strip()]

            return output_queries

        except (ValueError, SyntaxError) as e:
            self.logger.warning(
                f"Fallback to line splitting for output parsing. Error: {str(e)}"
            )
            # Remove any surrounding brackets and quotes, and trailing '\']"' characters
            clean_text = re.sub(r"^\[?'?\[?|'?\]?\]?$|\\'\]\"$", "", output_text)
            output_queries = [
                q.strip().strip("'").strip('"')
                for q in clean_text.split(",")
                if q.strip()
            ]

            if not output_queries:
                raise ValueError("No valid queries found after fallback parsing")

            return [process_string(q) for q in output_queries]
