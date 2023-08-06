from transformers import pipeline
generator = pipeline('text-generation')

prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

result = generator(prompt, max_length=100, num_return_sequences=5)

print(result)
