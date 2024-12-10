import ast
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from transformers import AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from seqeval.metrics import precision_score, recall_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Map IDs back to label names
    label_map = {v: k for k, v in label2id.items()}
    true_labels = [
        [label_map[l] for l, p in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]
    predictions = [
        [label_map[p] for l, p in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]

    # Flatten lists for calculation
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    predictions_flat = [item for sublist in predictions for item in sublist]

    # Calculate NER-specific metrics
    precision = precision_score(true_labels_flat, predictions_flat, average="weighted")
    recall = recall_score(true_labels_flat, predictions_flat, average="weighted")
    f1 = f1_score(true_labels_flat, predictions_flat, average="weighted")

    return {"precision": precision, "recall": recall, "f1": f1}

def compute_label_distribution(dataset, label2id):
    label_counts = Counter()
    for example in dataset:
        labels = example["labels"]  # Assuming the dataset has a "labels" key
        for label in labels:
            if label != -100:  # Ignore special tokens like padding
                label_counts[label] += 1

    # Map to label names if needed
    return {label: count for label, count in label_counts.items()}

def compute_class_weights(label_counts, label2id):
    total = sum(label_counts.values())
    weights = {k: total / (len(label_counts) * label_counts.get(k, 1)) for k in label2id.values()}
    return torch.tensor([weights[id] for id in range(len(label2id))], dtype=torch.float)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        return self.loss_fn(logits, labels)

class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional information during training."""
        if logs is not None:
            print(f"Step: {state.global_step} | Logs: {logs}")

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training has begun...")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} has ended.")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training is complete.")

def preprocess_data_with_label_mapping(row, save_mapping_path="label_mapping.json"):
    global label2id
    tokens, labels = [], []
    content = row["line"]  # Line text content
    entities = row["entities"]  # List of entity dictionaries

    # Create a mapping for character-level labels
    label_map = ["O"] * len(content)
    for entity in entities:
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_type = entity["type"]

        if f"B-{entity_type}" not in label2id:
            label2id[f"B-{entity_type}"] = len(label2id)
            label2id[f"I-{entity_type}"] = len(label2id)

        label_map[entity_start] = f"B-{entity_type}"
        for i in range(entity_start + 1, entity_end):
            label_map[i] = f"I-{entity_type}"

    # Tokenize the content into words and assign labels
    current_position = 0
    for token in content.split():
        tokens.append(token)
        token_label = "O"  # Default label
        token_start = current_position
        token_end = current_position + len(token)

        # Check if token matches an entity label in the map
        if any(label != "O" for label in label_map[token_start:token_end]):
            token_label = label_map[token_start]

        # Convert label to integer using the label2id mapping
        label_id = label2id.get(token_label, 0)
        labels.append(label_id)
        current_position += len(token) + 1  # Account for spaces

    # Save the label mapping to a JSON file
    with open(save_mapping_path, "w") as file:
        json.dump(label2id, file, indent=4)

    return {"tokens": tokens, "labels": labels}

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        padding="max_length",  # Ensure all sequences in a batch have the same length
        is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()  # Map tokens back to their original word
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:  # Special tokens like [CLS] and [SEP]
            labels.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first subword of each token
            labels.append(example["labels"][word_idx])
        else:  # For other subwords of the same word, use -100 to ignore them in loss calculation
            labels.append(-100)
        previous_word_idx = word_idx

    if len(labels) != len(word_ids):
        print(f"Warning: Mismatch in labels and word IDs. Labels: {len(labels)}, Word IDs: {len(word_ids)}")

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

data_path = "results/processed_data.csv"
df = pd.read_csv(data_path)
df["entities"] = df["entities"].apply(ast.literal_eval) # Deserialize entities from JSON-like format

# Apply preprocessing to each row
label2id = {"O": 0}  # Initialize the label-to-integer mapping
processed = df.apply(preprocess_data_with_label_mapping, axis=1)
processed = processed.tolist()  # Convert processed rows to a list
valid_data = [row for row in processed if isinstance(row, dict)]  # Filter valid rows
print("Sample preprocessed data:", valid_data[:2])
dataset = Dataset.from_list(valid_data)  # Create Hugging Face Dataset

# compute weight for balanced training
label_distribution = compute_label_distribution(dataset, label2id)
class_weights = compute_class_weights(label_distribution, label2id)
if len(class_weights) != len(label2id):
    raise ValueError("Mismatch between class weights and label2id size.")

# Tokenize data for Hugging Face model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply tokenization with mapping
try:
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=["tokens", "labels"])
    print("Mapping successful without batching!")
except Exception as e:
    print(f"Error during non-batched mapping: {e}")

# Split dataset into training and test sets
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Load model for token classification
class CustomModelForTokenClassification(AutoModelForTokenClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.loss_fn = WeightedCrossEntropyLoss(class_weights)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, class_weights=None, **kwargs):
        # Load the pre-trained model using the parent class's method
        model = super(CustomModelForTokenClassification, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        # Initialize the custom weighted loss
        model.loss_fn = WeightedCrossEntropyLoss(class_weights)
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits, "hidden_states": outputs.hidden_states, "attentions": outputs.attentions}

# Load the configuration
config = AutoConfig.from_pretrained(model_name, num_labels=len(label2id))

# Instantiate the model with pre-trained weights and custom loss
model = CustomModelForTokenClassification.from_pretrained(
    model_name,
    config=config,
    class_weights=class_weights  # Pass the class weights here
)
# Define training arguments
fp16 = torch.cuda.is_available()
if not fp16:
    print("GPU not available, disabling mixed precision training.")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    logging_first_step=True,  # Log the first step
    report_to="none",  # Disable sending logs to external tools like TensorBoard or WandB
    log_level="debug",  # Set log level to debug
    disable_tqdm=False,  # Ensure tqdm progress bar is visible
    fp16=fp16,  # Enable mixed precision training for reduced memory usage
    save_total_limit=2,  # Limit checkpoint saves to conserve disk space
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[DebugCallback()] # Add your custom callback
)

print("Evaluating the initial model performance...")
initial_metrics = trainer.evaluate()  # Evaluate the model before training
print(f"Initial Metrics: {initial_metrics}")

print("model training start:")
# Train the model
trainer.train()

# Save the best model
trainer.save_model("trained_ner_model")
print("Model saved to 'trained_ner_model'")
