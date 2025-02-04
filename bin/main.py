from bin.data_generation.construct_prompt import load_prompt

# Load the prompt
prompt = load_prompt(id="attribute_controlled_prompt")
print(prompt.prompt)