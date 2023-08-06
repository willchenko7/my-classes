from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TrainingArguments, Trainer
from torchinfo import summary

#load glue dataset
raw_dataset = load_dataset('glue', 'sst2')

checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)

tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    'my_trainer',
    evaluation_strategy='epoch',
    num_train_epochs=1,
    save_strategy='epoch'
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

params_before = []
for param in model.parameters():
    params_before.append(param.clone())

metric = load_metric('glue', 'sst2')
metric.compute(predictions=[1,0,1],references=[1,0,0])

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model('my_trainer')