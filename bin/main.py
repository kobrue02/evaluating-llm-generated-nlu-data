from bin.data_generation.construct_prompt import load_prompt
from bin.data_generation.generate_data import DataGenerationModel
from bin.framework.framework import Framework

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

# Load the prompt
prompt = load_prompt("one_shot_simple")

# Generate the data
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
phi = DataGenerationModel(model=model, tokenizer=tokenizer)

# Generate the data
data = phi.build_dataset_from_intents(
    "few_shot_simple",
    intents=["ac_on", "ac_off", "music_on", "music_off"],
    samples_per_intent=5,
)
data = data.to_data_frame(columns=["text", "intent"])
print(data)

golden_data = pd.DataFrame(
    {
        "text": [
            "Mach die Klima an",
            "Klimaanlage einschalten",
            "Mach Klima aus",
            "Klima aus",
            "Musik anmachen",
            "Musik spielen",
            "Musik aus",
            "Musik ausmachen",
        ],
        "intent": [
            "ac_on",
            "ac_on",
            "ac_off",
            "ac_off",
            "music_on",
            "music_on",
            "music_off",
            "music_off",
        ],
    }
)
print(golden_data)

# Apply the framework
framework = Framework()
result = framework.apply_framework_to_datasets(golden_data, data)
for intent in result:
    print(intent)
