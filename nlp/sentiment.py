from transformers import pipeline
classifier = pipeline('sentiment-analysis')

text = "This is such a great movie!"
result = classifier(text)
print(result[0]['label'])