
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
    def __init__(self, *args, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def generate_synthetic_data(self, prompt: Prompt) -> DataSet:
        synthetic_data = DataSet()
        self.logger.info("Initializing pipeline")
        try:
            pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
            raise

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        messages = list(prompt)
        self.logger.info("Generating text")
        try:
            output = pipe(messages, **generation_args)
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            raise

        self.logger.info("Parsing output")
        try:
            output_queries = self._parse_output(output[0]["generated_text"])
        except (ValueError, SyntaxError, TypeError) as e:
            self.logger.error(f"Error parsing generated queries: {e}")
            raise MalformedOutputError(f"Error parsing generated queries: {e}")

        synthetic_data.extend(
            output_queries, labels=[prompt.intent] * len(output_queries)
        )
        return synthetic_data

    def build_dataset_from_intents(
        self, prompt_id: str, intents: list[str], samples_per_intent: int = 10
    ) -> DataSet:
        synthetic_data = DataSet()
        for intent in tqdm(intents, desc="Processing intents"):
            self.logger.info(f"Generating data for intent: {intent}")
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
                except MalformedOutputError as e:
                    self.logger.warning(f"MalformedOutputError for {intent}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for {intent}: {e}")
                    continue

            synthetic_data += DataSet(list(unique_samples.keys()))

        return synthetic_data

# ... (rest of the code remains the same)
