from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#tokenize string
text = "Hello world"
tokens = tokenizer.tokenize(text)

#convert tokens to ids
ids = tokenizer.convert_tokens_to_ids(tokens)

#convert ids to tokens
tokens = tokenizer.convert_ids_to_tokens(ids)

#decode tokens to string
new_text = tokenizer.decode(ids)

#encode string to tokens
tokens = tokenizer.encode(text)

data = [
    "I like cats",
    "Do you like cats?"
]

tokenizer(data, padding=True, truncation=True, return_tensors="pt")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

outputs = model(**model_inputs)
