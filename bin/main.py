from utils import *
from generate_data import generate_synthetic_data
from framework import (
    calculate_perplexity,
    calculate_coherence,
    task_specific_performance,
    distinct_n,
)
from apply_framework import evaluate_synthetic_data
from comparative_analysis import compare_strategies

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Example prompt
prompt = ""
real_data = DataSet()

llama_data = generate_synthetic_data(prompt, llama_model, llama_tokenizer)
phi_data = generate_synthetic_data(prompt, phi_model, phi_tokenizer)

metrics = {
    "perplexity": calculate_perplexity,
    "diversity": lambda x, _: distinct_n(x, 3),
    "coherence": calculate_coherence,
    "task_performance": task_specific_performance,
}

llama_results = evaluate_synthetic_data(llama_data, real_data, metrics)
phi_results = evaluate_synthetic_data(phi_data, real_data, metrics)

compare_strategies({"Llama": llama_results, "Phi": phi_results})
