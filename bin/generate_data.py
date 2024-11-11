from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import DataSet

def generate_synthetic_data(prompt, model, tokenizer, num_samples=100) -> DataSet:
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
    synthetic_data = DataSet()
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        synthetic_data.append(tokenizer.decode(outputs[0]))
    return synthetic_data

if __name__ == "__main__":
    # Load models
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

    # Example prompt
    prompt = "Generate a natural language understanding example:"

    llama_data = generate_synthetic_data(prompt, llama_model, llama_tokenizer)
    phi_data = generate_synthetic_data(prompt, phi_model, phi_tokenizer)