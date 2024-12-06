from transformers import pipeline
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
        
        pipe = pipeline( 
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        ) 

        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        messages = list(prompt)
        output = pipe(messages, **generation_args)

        try:
            output_queries = ast.literal_eval(output[0]['generated_text'])
            output_queries = output_queries
        except (ValueError, SyntaxError, TypeError) as e:
            raise MalformedOutputError(f"Error parsing generated queries: {e}")
        
        synthetic_data.extend(
            output_queries,
            labels=[prompt.intent]*len(output_queries)
            )
        return synthetic_data

    def build_dataset_from_intents(self, prompt_id: str, intents: list[str], samples_per_intent: int = 10) -> DataSet:
        """
        Generate synthetic data from a list of intents.

        Args:
            intents (list[str]): The intents to generate data from.
        
        Returns:
            DataSet: The synthetic data.
        """
        synthetic_data = DataSet()
        for intent in intents:
            prompt = load_prompt(id=prompt_id, intent=intent, num_samples=samples_per_intent)
            data = self.generate_synthetic_data(prompt)
            synthetic_data += data
        return synthetic_data


