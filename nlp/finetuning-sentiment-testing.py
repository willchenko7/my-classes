from transformers import pipeline

newmodel = pipeline('text-classification', model='my_trainer/checkpoint-8419', device=0)

newmodel('This movie is great!')

#fix config file
config_path = 'my_trainer/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

config['id2label'] = {0: 'NEGATIVE', 1: 'POSITIVE'}

with open(config_path, 'w') as f:
    json.dump(config, f)

newmodel = pipeline('text-classification', model='my_trainer', device=0)

newmodel('This movie is great!')

params_after = []
for param in model.parameters():
    params_after.append(param.clone())

for param1, param2 in zip(params_before, params_after):
    print(np.sum(np.abs(param1 - param2)))