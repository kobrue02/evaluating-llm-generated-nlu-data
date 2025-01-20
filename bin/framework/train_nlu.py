from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from bin.utils.types import DataSet

# Prepare your data
train_data = DataSet.from_dict({
    "data": ["book a flight", "what's the weather", ...],
    "labels": [0, 1, ...]  # Encoded intents
})

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_data["label"])))
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_data = train_data.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./my_nlu_model")
tokenizer.save_pretrained("./my_nlu_model")
