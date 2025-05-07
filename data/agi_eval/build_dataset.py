import ujson as json
import os

intro_dict = {
        'lsat-ar':
        'LSAT Analytical Reasoning',
        'lsat-lr':
        'LSAT Logical Reasoning',
        'lsat-rc':
        'LSAT Reading Comprehension',
        'logiqa-en':
        'Logic Reasoning',
        'sat-math':
        'SAT Math',
        'sat-en':
        'SAT English',
        'sat-en-without-passage':
        'SAT English',
        'aqua-rat':
        'AQUA-RAT',
        'math':
        'Math',
    }

file_path_list = os.listdir('v1/')
data = []
for file_path in file_path_list:
    if file_path.endswith(".jsonl"):
        with open('v1/' + file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['question_type'] = intro_dict[file_path.split('.')[0]]
                data.append(item)
        

fw = open("test.jsonl", "w")
for item in data:
    question = item['question']
    if item['passage']:
        question = item['passage'] + '\n\n' + item['question']

    if item['options']:
        # "options": ["(A)Mean", "(B)Median", "(C)Range", "(D)Standard deviation"]
        # Remove the "(A)", and change it to "A. Mean"
        for i in range(len(item['options'])):
            if item['options'][i][0] == '(' and item['options'][i][2] == ')':
                item['options'][i] = item['options'][i][1] + '. ' + item['options'][i][3:]
        options = '\n'.join(item['options'])
        question += '\n\n' + options

    item['question'] = question

    if item['label']:
        item['answer'] = item['label']

    fw.write(json.dumps(item, ensure_ascii=False) + "\n")