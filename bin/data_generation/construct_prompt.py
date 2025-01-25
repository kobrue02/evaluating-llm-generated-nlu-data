"""
Functions for loading prompts.

Author: Konrad Brüggemann
"""

from typing import List, Dict
from string import Formatter

import json


def extract_variable_names(template):
    """
    Extract variable names from a template.
    Args:
        template (str): The template.
    Returns:
        List[str]: The variable names.
    """
    return [
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    ]


class Prompt:
    """
    A class to represent a prompt.
    """

    def __init__(
        self, prompt: List[Dict], intent: str = None, examples: List[str] = None
    ):
        """
        Initialize a prompt.
        Args:
            prompt (List[Dict]): The prompt.
            intent (str): The intent of the prompt.
            examples (List[str]): Examples to append to the prompt.
        """
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


def load_prompt(
    path: str = None,
    id: int = None,
    generated_queries: List[str] = None,
    **kwargs,
) -> Prompt:
    """
    Load a prompt from a file.
    Args:
        path (str): The path to the file.
        id (int): The ID of the prompt to load.
        generated_queries (List[str]): Previously generated queries to append to the prompt.
    Returns:
        Prompt: The loaded prompt.
    """
    if path:
        try:
            with open(path, "r", encoding="utf-8") as file:
                prompt = json.load(file)
        except FileNotFoundError:
            return Prompt([], intent=kwargs.get("intent"))
    elif id:
        try:
            with open(f"bin/prompts/{id}.json", "r", encoding="utf-8") as file:
                prompt = json.load(file)
        except FileNotFoundError:
            return Prompt([], intent=kwargs.get("intent"))
    else:
        raise ValueError("Either path or id must be provided.")

    if isinstance(prompt, str):
        return Prompt(
            [{"role": "user", "content": prompt}],
            intent=kwargs.get("intent")
        )
    else:
        for i, message in enumerate(prompt):
            if message.get("role") == "user":
                for var in extract_variable_names(message["content"]):
                    if var in kwargs:
                        message["content"] = message["content"].format(**kwargs)
            if message.get("role") == "assistant":
                prompt[i]["content"] = str(prompt[i]["content"])

    if generated_queries:
        if prompt[-1]["role"] == "user":
            if isinstance(prompt[-1]["content"], tuple):
                prompt[-1]["content"] = prompt[-1]["content"][
                    0
                ]  # Take the first element if it's a tuple
            prompt[-1]["content"] = str(prompt[-1]["content"])  # Ensure it's a string
            prompt[-1]["content"] += "\n\nPreviously generated queries:\n" + "\n".join(
                map(str, generated_queries)
            )
        else:
            prompt.append(
                {
                    "role": "user",
                    "content": "Previously generated queries:\n"
                    + "\n".join(map(str, generated_queries)),
                }
            )

    return Prompt(prompt, intent=kwargs.get("intent"))


if __name__ == "__main__":
    prompt = load_prompt(
        path="bin/prompts/chain_of_thought_simple.json", intent="ac_on", num_samples=10, examples=["example1", "example2"]
    )
    print(prompt)
