from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import DataSet

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
        instruction_prompt = f"Generate {num_samples} synthetic NLU training queries similar to the following query: `{prompt}`. Return the queries in a list."
        
        # call the model to generate synthetic data
        input_ids = tokenizer.encode(instruction_prompt, return_tensors="pt", max_length=1024, truncation=True)
        output = model.generate(input_ids, max_length=128, num_return_sequences=1, early_stopping=True)
        synthetic_data.append(tokenizer.decode(output[0], skip_special_tokens=True), label="intent")
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