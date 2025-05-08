# This script fine-tunes a multiple-choice model for the Word Puzzle (WP) task

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForMultipleChoice, Trainer, TrainingArguments,
                          get_scheduler, DataCollatorForMultipleChoice)
import evaluate

# Set task and model
TASK = "WP"
MODEL_NAME = "FacebookAI/roberta-large"

# Loading data
train_data = np.load("/scratch/cs529304/project/data/WP-train.npy", allow_pickle=True)
test_data = np.load("/scratch/cs529304/project/data/WP_test_labeled.npy", allow_pickle=True)

# Create output directory
DATE = pd.to_datetime("today").strftime("%Y_%m_%d_%H_%M")
RUN_DIR = f"run_{TASK}_{DATE}"
os.makedirs(RUN_DIR, exist_ok=True)

# Convert .npy to HuggingFace Dataset
def convert_to_dataset(numpy_array, split):
    df = pd.DataFrame(numpy_array.tolist())

    for col in ['distractor1', 'distractor2', 'distractor(unsure)']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df['label'] = df['label'].astype(int)

    return Dataset.from_pandas(df)


train_dataset = convert_to_dataset(train_data, "train")
test_dataset = convert_to_dataset(test_data, "test")

# Preprocessing

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["question"]]
    second_sentences = examples["choice_list"]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    tokenized = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Filter and split subsets
original = tokenized_train.filter(lambda x: "_SR" not in x["id"] and "_CR" not in x["id"])
semantic = tokenized_train.filter(lambda x: "_SR" in x["id"])
context = tokenized_train.filter(lambda x: "_CR" in x["id"])

def split_dataset(ds):
    temp = ds.train_test_split(test_size=0.3, shuffle=False)
    tv = temp["test"].train_test_split(test_size=0.5, shuffle=False)
    return DatasetDict({"train": temp["train"], "valid": tv["train"], "test": tv["test"]})

original_ds = split_dataset(original)
semantic_ds = split_dataset(semantic)
context_ds = split_dataset(context)

train_ds = concatenate_datasets([original_ds["train"], semantic_ds["train"], context_ds["train"]]).shuffle(seed=42)
valid_ds = concatenate_datasets([original_ds["valid"], semantic_ds["valid"], context_ds["valid"]]).shuffle(seed=42)

# Final dataset
final_ds = DatasetDict({"train": train_ds, "valid": valid_ds})

def format_dataset(ds):
    ds = ds.rename_column("label", "labels")
    ds = ds.remove_columns(['id', 'question', 'answer', 'distractor1', 'distractor2', 'distractor(unsure)', 'choice_list', 'choice_order'])
    ds.set_format("torch")
    return ds

tokenized_final = format_dataset(final_ds)

# Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Model and training setup
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

training_args = TrainingArguments(
    output_dir=os.path.join(RUN_DIR, "output"),
    evaluation_strategy="steps",
    eval_steps=20,
    logging_steps=20,
    logging_strategy="steps",
    learning_rate=3e-5,
    num_train_epochs=3,
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

# Save
model.save_pretrained(os.path.join(RUN_DIR, "model"))

# Final Evaluation on the test set
test_dataset_formatted = format_dataset(tokenized_test)
metrics = trainer.evaluate(test_dataset_formatted)

# Save and print metrics
results_path = os.path.join(RUN_DIR, "test_results.txt")
with open(results_path, "w") as f:
    for key, value in metrics.items():
        line = f"{key}: {value}\n"
        print(line.strip())
        f.write(line)

# Also save in CSV format
df = pd.DataFrame([metrics])
df["model"] = MODEL_NAME
df["timestamp"] = DATE
df.to_csv(os.path.join(RUN_DIR, "results.csv"), index=False)

print("\nEvaluation complete. Results saved to:", results_path)
