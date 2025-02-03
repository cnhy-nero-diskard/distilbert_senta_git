"""WARNING: THIS TRAINING PIPELINE FOR SOME REASON ONLY SPITS OUT LABEL_0. Don't use this"""

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
dataset = dataset.map(
    lambda examples: {'label': label_encoder.fit_transform(examples['label'])},
    batched=True
)

# Function to reduce dataset size by 70% while maintaining label balance
def reduce_dataset_size(dataset, reduction_factor=0.4):
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
reduced_dataset = reduce_dataset_size(dataset, reduction_factor=0.1)

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
    output_dir='./results',
    evaluation_strategy="epoch",
    fp16=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir='./logs',
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

# Save the model and tokenizer
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')