from transformers import pipeline

mlm = pipeline('fill-mask')

result = mlm("This is a <mask> example.")

print(result)