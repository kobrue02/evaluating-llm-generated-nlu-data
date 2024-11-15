from typing import Union
from transformers import pipeline
from utils import DataSet

import ast
import json
import torch

torch.random.manual_seed(0)

class DataGenerationModel:

    def __init__(self, *args, model=None, tokenizer=None):
        """
        Initialize the data generation model.
        Args:
            model (transformers.PreTrainedModel): The language model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_synthetic_data(self, prompt: list=None, query: str=None, model=None, tokenizer=None, num_samples=100) -> DataSet:
        """
        Generate synthetic data using the model.

        Args:
            prompt (list): The prompt to generate data from.
            query (str): The query to generate data for.
            model (transformers.PreTrainedModel): The language model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
            num_samples (int): The number of samples to generate.
        
        Returns:
            DataSet: The synthetic data.
        """
        if not model:
            model = self.model
        
        if not tokenizer:
            tokenizer = self.tokenizer

        synthetic_data = DataSet()
        
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer
        ) 

        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        if not prompt:
            messages = [ 
                {"role": "system", "content": "You are an NLU expert, with a focus on NLU data generation."}, 
                {"role": "user", "content": "Can you generate 5 queries for the intent `ac_on`?"}, 
                {"role": "assistant", "content": "['Turn on the AC in the back of the car.', 'AC on', 'Put on the air con', 'Can you turn on AC?']"},
                {"role": "user", "content": f"How about {num_samples} queries for the intent {query}? Please return the queries in Python list format."}, 
            ]
        else:
            messages = prompt

        output = pipe(messages, **generation_args)
        try:
            output_queries = ast.literal_eval(output[0]['generated_text'])
        except ValueError:
            output_queries = output[0]['generated_text']
        synthetic_data.append(output_queries, label="intent")

        return synthetic_data
    
def load_prompt(path: str=None,  id: int=None, **kwargs) -> Union[list[dict], str]:
    """
    Load a prompt from a file.
    Args:
        path (str): The path to the file.
        id (int): The ID of the prompt to load.
    Returns:
        list: The prompt.
    """
    if path:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                prompt = json.load(file)
        except FileNotFoundError:
            return []
    elif id:
        try:
            with open(f'prompts/{id}.json', 'r', encoding='utf-8') as file:
                prompt = json.load(file)
        except FileNotFoundError:
            return []
    else:
        raise ValueError("Either path or id must be provided.")

    if isinstance(prompt, str):
        return prompt
    else:
        for i, message in enumerate(prompt):
            if message.get("role") == "user":
                prompt[i]["content"] = prompt[i]["content"].format(**kwargs)
                    
    return prompt


if __name__ == "__main__":
    prompt = load_prompt(path="prompts/chat_template_basic.json", query="ac_on", num_samples=5)
    print(prompt)

