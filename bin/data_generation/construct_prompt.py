"""
Functions for loading prompts.

Author: Konrad Brüggemann
"""

from typing import Union, List, Dict

import json


class Prompt:
    """
    A class to represent a prompt.
    """
    def __init__(self, prompt: List[Dict], intent: str=None):
        self.prompt = prompt
        self.intent = intent
    
    def __str__(self):
        return json.dumps(self.prompt, indent=4)
    
    def __repr__(self):
        return str(self.prompt)
    
    def __len__(self):
        return len(self.prompt)
    
    def __iter__(self):
        return iter(self.prompt)


def load_prompt(path: str=None,  id: int=None, **kwargs) -> Prompt:
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
                    
    return Prompt(prompt, intent=kwargs.get("intent"))


if __name__ == "__main__":
    prompt = load_prompt(path="bin/prompts/chain_of_thought_simple.json", intent="ac_on", num_samples=10)
    print(prompt)