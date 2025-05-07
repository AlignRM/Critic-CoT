import openai
import os
os.environ["AZURE_DEPLOYMENT"] = 'gpt4-turbo'

import argparse
from os import error
import re
from time import sleep
import random
from tqdm import tqdm
import json
from utils import parse_correction_and_changed_answer, parse_label
import sys 
sys.path.append("..") 
sys.path.append(".") 
from grading.grader import grade_answer

from multiprocessing import Pool
import copy
import time

random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4-turbo')
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--num_shot', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sample', type=int, default=1)
parser.add_argument('--stop', type=str, default=None, nargs='+')
parser.add_argument('--max_try', type=int, default=200)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--max_tokens', type=int, default=4096)
parser.add_argument('--max_instance', type=int, default=99999)
parser.add_argument('--token', type=str, default='')
args = parser.parse_args()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": '''Remove any phrases like "reference solution" or "hint".\n\n-----\n\nStep 1: The problem statement is correctly understood.\nConclusion: Step 1 is correct.\n\nStep 2: The approach to simplify the numerator is a valid initial thought process.\nConclusion: Step 2 is correct.\n\nStep 3: The calculation of $235^2 = 55225$ is accurate.\nConclusion: Step 3 is correct.\n\nStep 4: The calculation of $221^2 = 48441$ is accurate.\nConclusion: Step 4 is correct.\n\nStep 5: Subtracting $48441$ from $55225$ to get $6784$ is correct.\nConclusion: Step 5 is correct.\n\nStep 6: The decision to divide the result by 14 is in line with the problem's instructions.\nConclusion: Step 6 is correct.\n\nStep 7: The division of $6784$ by $14$ is where the mistake occurs. The correct division should result in $484$, but let's verify this calculation.\nConclusion: Step 7 is incorrect.\n<correction>\nStep 7 (Corrected): Correctly divide $6784$ by $14$.\n\\[ \\frac{6784}{14} = 484 \\]\n\nHowever, the mistake identified in the hint suggests a reevaluation of the calculation or the approach. The reference solution uses a factoring technique based on the difference of squares formula, which was not considered in the initial attempt. Let's apply the correct approach based on the reference solution:\n\nThe correct approach is to recognize that $235^2 - 221^2$ can be factored as $(235 + 221)(235 - 221)$ using the difference of squares formula, $a^2 - b^2 = (a + b)(a - b)$.\n\n\\[ \\frac{235^2 - 221^2}{14} = \\frac{(235 + 221)(235 - 221)}{14} \\]\n\n\\[ = \\frac{456 \\cdot 14}{14} \\]\n\n\\[ = 456 \\]\n\nSo, the correct value of $\\displaystyle\\frac{235^2-221^2}{14}$ is $\\boxed{456}$.\n</correction>'''},
    {"role": "assistant", "content": '''Step 1: The problem statement is correctly understood.\nConclusion: Step 1 is correct.\n\nStep 2: The approach to simplify the numerator is a valid initial thought process.\nConclusion: Step 2 is correct.\n\nStep 3: The calculation of $235^2 = 55225$ is accurate.\nConclusion: Step 3 is correct.\n\nStep 4: The calculation of $221^2 = 48441$ is accurate.\nConclusion: Step 4 is correct.\n\nStep 5: Subtracting $48441$ from $55225$ to get $6784$ is correct.\nConclusion: Step 5 is correct.\n\nStep 6: The decision to divide the result by 14 is in line with the problem's instructions.\nConclusion: Step 6 is correct.\n\nStep 7: The division of $6784$ by $14$ is where the mistake occurs. The correct division should result in $484$, but let's verify this calculation.\nConclusion: Step 7 is incorrect.\n<correction>\nStep 7 (Corrected): Correctly divide $6784$ by $14$.\n\\[ \\frac{6784}{14} = 484 \\]\n\nLet's apply the correct approach:\n\nThe correct approach is to recognize that $235^2 - 221^2$ can be factored as $(235 + 221)(235 - 221)$ using the difference of squares formula, $a^2 - b^2 = (a + b)(a - b)$.\n\n\\[ \\frac{235^2 - 221^2}{14} = \\frac{(235 + 221)(235 - 221)}{14} \\]\n\n\\[ = \\frac{456 \\cdot 14}{14} \\]\n\n\\[ = 456 \\]\n\nSo, the correct value of $\\displaystyle\\frac{235^2-221^2}{14}$ is $\\boxed{456}$.\n</correction>'''},
]
with open(args.src, 'r') as f:
    instance_list = f.readlines()[:args.max_instance]
instance_list = [json.loads(instance) for instance in instance_list]

cnt_previous = 0
try:
    with open(args.tgt, 'r') as f:
        cnt_previous = len(f.readlines())
except:
    pass

def infer_stream(messages, temperature=1, max_tokens=1000, stop=None, top_p=1.0):
    client = openai.OpenAI(api_key=args.token)
    response = client.chat.completions.create(model=args.model, messages=messages, temperature=temperature, stop=stop, max_tokens=max_tokens, top_p=top_p, stream=True)

    completion_text = ''
    # iterate through the stream of events
    try:
        for event in response:
            try:
                event_text = event.choices[0].delta.content  # extract the text
                if event_text is None:
                    event_text = ''
            except:
                event_text = ''
            completion_text += event_text  # append the text
            print(event_text, end='')  # print the delay and text
    except Exception as e:
        print(e)
    return completion_text


def infer_instance(instance):
    all_critics = []
    all_refinements = []

    for old_critic, old_pred_label, old_refine_answer, old_refinement in zip(instance["critics"], instance["pred_labels"], instance["refine_answers"], instance["refinements"]):
        old_critic_lower = old_critic.lower()
        if "hint" not in old_critic_lower and "reference" not in old_critic_lower:
            print("Skip")
            all_critics.append(old_critic)
            all_refinements.append(old_refinement)
            continue

        prompt = f'''For the following text, remove any phrases like "reference solution" or "hint", and keep all the other content. Do not miss the "<correction>" and "</correction>" label is exist in the text. Do not response anything else.\n\n-----\n\n{old_critic}'''
        # print('\n', prompt)
        new_messages = messages + [{"role": "user", "content": prompt}]
        success = False
        for _ in range(16):
            new_critic = infer_stream(new_messages, model=args.model, max_try=args.max_try, temperature=args.temperature, stop=args.stop, max_tokens=args.max_tokens, top_p=args.top_p)
            # print(new_critic)
            new_pred_label = parse_label(new_critic, len(old_pred_label))
            new_refinement, new_refine_answer = parse_correction_and_changed_answer(new_critic)
            _, old_refine_answer = parse_correction_and_changed_answer(old_critic)
            if new_pred_label == old_pred_label and new_refine_answer == old_refine_answer:
                all_critics.append(new_critic)
                all_refinements.append(new_refinement)
                success = True
                break
            else:
                print('Retry')
                print('old_critic\n', old_critic)
                print(new_pred_label, old_pred_label)
                print(new_refine_answer, old_refine_answer)
                sleep(10)
        if not success:
            all_critics.append(old_critic)
            all_refinements.append(old_refinement)

    instance["critics"] = all_critics
    instance["refinements"] = all_refinements


    return instance

with open(args.tgt, 'a') as fw:
    with Pool(args.batch_size) as pool:
        for new_instance in tqdm(pool.imap(infer_instance, instance_list[cnt_previous:]), total=len(instance_list[cnt_previous:])):
            print(json.dumps(new_instance), file=fw)
            fw.flush()
