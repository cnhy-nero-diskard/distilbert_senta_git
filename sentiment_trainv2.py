# This script fine-tunes a DistilBERT model for sentiment analysis on a custom dataset.
# The script performs the following steps:
# 1. Load the dataset from a JSON file.
# 2. Initialize the tokenizer for the DistilBERT model.
# 3. Tokenize the dataset.
# 4. Encode the labels using a label encoder.
# 5. Reduce the dataset size by a specified reduction factor while maintaining label balance.
# 6. Split the dataset into training, validation, and test sets.
# 7. Initialize the DistilBERT model for sequence classification.
# 8. Define training arguments for the Trainer.
# 9. Define a function to compute evaluation metrics.
# 10. Initialize the Trainer with the model, training arguments, datasets, tokenizer, and metrics function.
# 11. Train the model.
# 12. Evaluate the model on the test set.
# 13. Save the trained model and tokenizer.
# Functions:
# - tokenize_function(examples): Tokenizes the input examples using the tokenizer.
# - reduce_dataset_size(dataset, reduction_factor=0.7): Reduces the size of the dataset by a specified reduction factor while maintaining label balance.
# - compute_metrics(eval_pred): Computes evaluation metrics (accuracy) for the model predictions.
# Usage:
# Run the script to fine-tune the DistilBERT model on the custom dataset and evaluate its performance.
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate
from datasets import Dataset


# Load the dataset
dataset = load_dataset('json', data_files='./combined_data.json')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Initialize a label encoder
label_encoder = LabelEncoder()

# Fit and transform the labels
all_labels = dataset['train']['label']
label_encoder.fit(all_labels)

# Transform labels using a single mapping function
dataset = dataset.map(lambda examples: {'label': label_encoder.transform(examples['label'])}, batched=True)
print(f"CLASSES --------------------> \n {label_encoder.classes_}")

# Function to reduce dataset size by 70% while maintaining label balance
def reduce_dataset_size(dataset, reduction_factor=0.7):
    """
    Reduces the size of the dataset by a specified reduction factor while maintaining label balance.

    Parameters:
    dataset (dict): A dictionary containing the dataset with a 'train' key. The 'train' key should map to a dataset object that has 'text' and 'label' fields.
    reduction_factor (float): The factor by which to reduce the dataset size. Default is 0.7, meaning the dataset will be reduced by 70%.

    Returns:
    dict: A dictionary containing the reduced dataset with 'text' and 'label' fields.

    Example:
    >>> dataset = {
    ...     'train': {
    ...         'text': ['sample1', 'sample2', 'sample3', 'sample4'],
    ...         'label': [0, 1, 0, 1]
    ...     }
    ... }
    >>> reduced_dataset = reduce_dataset_size(dataset, reduction_factor=0.5)
    >>> print(reduced_dataset)
    {'text': ['sample1', 'sample2'], 'label': [0, 1]}
    """
    # Get the unique labels
    unique_labels = set(dataset['train']['label'])
    
    # Initialize an empty list to store the reduced dataset
    reduced_data = {'text': [], 'label': []}
    
    for label in unique_labels:
        # Filter the dataset for the current label
        label_data = dataset['train'].filter(lambda example: example['label'] == label)
        
        # Calculate the number of samples to keep
        num_samples = int(len(label_data) * (1 - reduction_factor))
        
        # Shuffle and select the samples
        label_data = label_data.shuffle(seed=42).select(range(num_samples))
        
        # Append the selected samples to the reduced dataset
        reduced_data['text'].extend(label_data['text'])
        reduced_data['label'].extend(label_data['label'])
    
    # Convert the reduced data back to a Dataset object
    reduced_dataset = Dataset.from_dict(reduced_data)
    
    return reduced_dataset

# Reduce the dataset size
reduced_dataset = reduce_dataset_size(dataset, reduction_factor=0)

# Tokenize the reduced dataset
tokenized_datasets = reduced_dataset.map(tokenize_function, batched=True)

# Split the dataset
split_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']

# Further split the validation set into validation and test sets
val_test_split = val_dataset.train_test_split(test_size=0.5)
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-multilingual-cased',
    num_labels=3  # 3 classes: positive, neutral, negative
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./training/results',  
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir='./training/logs',  
    logging_steps=10,
)

# Load the accuracy metric
metric = evaluate.load("accuracy")

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(test_dataset)
print(results)

model.save_pretrained('./distilbert-multicased-sentimentTourism')
tokenizer.save_pretrained('./distilbert-multicased-sentimentTourism')