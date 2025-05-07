import json

for name in ['phase2_train']:
    file_name = f'{name}.jsonl'
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    data = [d for d in data if 'finish_reason' in d['label'] and d['label']['finish_reason'] in ["found_error", "solution"]]
    data = [d for d in data if d['label']['finish_reason'] in ["solution"] or len(d['question']['pre_generated_steps']) != len(d['label']['steps'])]


    output_name = f'{name}_filter.jsonl'
    with open(output_name, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
