from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, Dataset
import os



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)
    
    
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
    

synthetic = pd.read_csv('./synthetic_data/tweet_eval_train_abortion.csv').reset_index(drop=True)
synthetic_filtered = pd.read_csv('./synthetic_data/tweet_eval_train_abortion_filtered.csv').reset_index(drop=True)
original = load_dataset("tweet_eval", "abortion")
original_train = pd.DataFrame( original['train'] ).reset_index(drop=True)
original_validation = pd.DataFrame( original['validation'] ).reset_index(drop=True)
original_test = pd.DataFrame( original['test'] ).reset_index(drop=True)
combined_train = pd.concat([original_train, synthetic]).reset_index(drop=True)
combined_train = Dataset.from_pandas(combined_train)
combined_validation = Dataset.from_pandas(original_validation)
combined_test = Dataset.from_pandas(original_test)
original_and_synthetic = DatasetDict({"train": combined_train, "validation": combined_validation, "test": combined_test})

combined_train_filtered = pd.concat([original_train, synthetic_filtered]).reset_index(drop=True)
combined_train_filtered = Dataset.from_pandas(combined_train)
combined_validation_filtered = Dataset.from_pandas(original_validation)
combined_test_filtered = Dataset.from_pandas(original_test)
original_and_synthetic_filtered = DatasetDict({"train": combined_train, "validation": combined_validation, "test": combined_test})

print(original)
print(original_and_synthetic)

model_checkpoint = "roberta-base"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
tokenized_original = original.map(preprocess_function, batched=True)
tokenized_original_and_synthetic = original_and_synthetic.map(preprocess_function, batched=True)
tokenized_original_and_synthetic_filtered = original_and_synthetic_filtered.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


f1 = evaluate.load("f1")

id2label = {0: "none", 1: "against", 2:"favor"}
label2id = {"none": 0, "against": 1, "favor": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)



epoch = 30
learning_rate = 2e-5
training_args = TrainingArguments(
    output_dir="/xdisk/bethard/kbozler/output/directed-research-3/back-translation-w-synthetic-filtered",
    learning_rate=learning_rate,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=epoch,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_original_and_synthetic_filtered["train"],
    eval_dataset=tokenized_original_and_synthetic_filtered["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

trainer.train()