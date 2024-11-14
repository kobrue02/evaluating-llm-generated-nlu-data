from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import DataSet

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
        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("gpt2")

    def generate_synthetic_data(self, prompt, model=None, tokenizer=None, num_samples=100) -> DataSet:
        """
        Generate synthetic data using a language model.
        Args:
            prompt (str): The prompt to generate data from.
            model (transformers.PreTrainedModel): The language model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
            num_samples (int): The number of samples to generate.
        Returns:
            List[str]: The synthetic data samples.
        """
        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        synthetic_data = DataSet()
        
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=0 if torch.cuda.is_available() else -1,
        ) 

        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        messages = [ 
            {"role": "system", "content": "You are an NLU expert. Your task is to generate synthetic NLU training data for intent classification."}, 
            {"role": "user", "content": f"Generate {num_samples} synthetic data samples for the intent: {prompt}"},
        ] 

        output = pipe(tokenizer.apply_chat_template(messages), **generation_args) 
        synthetic_data.append(output[0]['generated_text'], label="intent")

        return synthetic_data

if __name__ == "__main__":
    # Load models
    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

    # Initialize data generation models
    phi = DataGenerationModel(model=phi_model, tokenizer=phi_tokenizer)

    # Generate synthetic data
    prompt = "Turn on the AC in the back of the car."
    phi_data = phi.generate_synthetic_data(prompt, num_samples=100)

    print(phi_data)