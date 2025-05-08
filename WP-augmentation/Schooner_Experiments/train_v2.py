# This script fine-tunes a multiple-choice model for the Word Puzzle (WP) task including paraphrased examples

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForMultipleChoice, Trainer, TrainingArguments,
                          DataCollatorForMultipleChoice)
import evaluate

# Set HuggingFace cache to scratch
os.environ["TRANSFORMERS_CACHE"] = "/scratch/cs529304/hf_cache"
os.environ["HF_HOME"] = "/scratch/cs529304/hf_cache"

# Set task and model
TASK = "WP_PARAPHRASE"
MODEL_NAME = "FacebookAI/roberta-large"

# Load full dataset (including _PARA entries)
data = np.load("/scratch/cs529304/project/data/WP-train.npy", allow_pickle=True)
test_data = np.load("/scratch/cs529304/project/data/WP_test_labeled.npy", allow_pickle=True)

# Create output directory
DATE = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")
RUN_DIR = f"v2_run_{TASK}_{DATE}"
os.makedirs(RUN_DIR, exist_ok=True)

# Convert to Dataset
def convert_to_dataset(numpy_array):
    df = pd.DataFrame(numpy_array.tolist())
    for col in ['distractor1', 'distractor2', 'distractor(unsure)']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df['label'] = df['label'].astype(int)
    return Dataset.from_pandas(df)

full_dataset = convert_to_dataset(data)
test_dataset = convert_to_dataset(test_data)

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["question"]]
    second_sentences = examples["choice_list"]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    tokenized = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}

tokenized_full = full_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Filter by ID
original = tokenized_full.filter(lambda x: "_SR" not in x["id"] and "_CR" not in x["id"] and "_PARA" not in x["id"])
semantic = tokenized_full.filter(lambda x: "_SR" in x["id"])
context = tokenized_full.filter(lambda x: "_CR" in x["id"])
paraphrase = tokenized_full.filter(lambda x: "_PARA" in x["id"])

def split_dataset(ds):
    temp = ds.train_test_split(test_size=0.3, shuffle=False)
    tv = temp["test"].train_test_split(test_size=0.5, shuffle=False)
    return DatasetDict({"train": temp["train"], "valid": tv["train"], "test": tv["test"]})

original_ds = split_dataset(original)
semantic_ds = split_dataset(semantic)
context_ds = split_dataset(context)
paraphrase_ds = split_dataset(paraphrase)

train_ds = concatenate_datasets([
    original_ds["train"],
    semantic_ds["train"],
    context_ds["train"],
    paraphrase_ds["train"]
]).shuffle(seed=42)

valid_ds = concatenate_datasets([
    original_ds["valid"],
    semantic_ds["valid"],
    context_ds["valid"],
    paraphrase_ds["valid"]
]).shuffle(seed=42)

# Final dataset
final_ds = DatasetDict({"train": train_ds, "valid": valid_ds})

def format_dataset(ds):
    ds = ds.rename_column("label", "labels")
    to_remove = ['id', 'question', 'answer', 'distractor1', 'distractor2', 'distractor(unsure)', 'choice_list', 'choice_order']
    ds = ds.remove_columns([c for c in to_remove if c in ds.column_names])
    ds.set_format("torch")
    return ds

tokenized_final = format_dataset(final_ds)

# Evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Model
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(RUN_DIR, "output"),
    eval_strategy="steps",
    eval_steps=20,
    logging_steps=20,
    logging_strategy="steps",
    learning_rate=3e-5,
    num_train_epochs=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to=None,
    save_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_final["train"],
    eval_dataset=tokenized_final["valid"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained(os.path.join(RUN_DIR, "model"))

# Evaluate on test set
test_dataset_formatted = format_dataset(tokenized_test)
metrics = trainer.evaluate(test_dataset_formatted)

# Save results
results_path = os.path.join(RUN_DIR, "test_results.txt")
with open(results_path, "w") as f:
    for key, value in metrics.items():
        line = f"{key}: {value}\n"
        print(line.strip())
        f.write(line)

# Save to CSV
df = pd.DataFrame([metrics])
df["model"] = MODEL_NAME
df["timestamp"] = DATE
df.to_csv(os.path.join(RUN_DIR, "results.csv"), index=False)

print("\nEvaluation complete. Results saved to:", results_path)
