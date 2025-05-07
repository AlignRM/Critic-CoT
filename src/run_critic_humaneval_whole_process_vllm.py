import argparse
from os import error
import re
from time import sleep
import random
random.seed()
from tqdm import tqdm
import json
import sys 
from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import copy

import logging

logging.basicConfig(level=logging.INFO)

sys.path.append("..") 
sys.path.append(".") 

from grading.code_grader import grade_answer
from utils import parse_label, parse_correction_and_changed_answer, extract_boxed_expressions_custom

from transformers import AutoTokenizer


random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--num_shot', type=int, default=0)
parser.add_argument('--num_return_sequences', type=int, default=1)
parser.add_argument('--best_of', type=int, default=8)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--max_tokens', type=int, default=3333)
parser.add_argument('--max_instance', type=int, default=10000)
parser.add_argument('--max_refine_depth', type=int, default=8)
parser.add_argument('--max_refine_restart', type=int, default=4)
parser.add_argument('--token', type=str, default='')
parser.add_argument('--mode', type=str, default='solve')
parser.add_argument('--repetition_penalty', type=float, default=1.1)

parser.add_argument("--address", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8002)
parser.add_argument("--address_file", type=str, default=None)
parser.add_argument("--request_batch_size", type=int, default=4)

args = parser.parse_args()

if args.address_file is None:
    url_list = [f"http://{args.address}:{args.port}/v1/chat/completions"]
else:
    # Read server address
    while True:
        try:
            with open(args.address_file, "r") as f:
                url_list = f.readlines()
                assert len(url_list) > 0
                url_list = [address.strip() for address in url_list]
                logging.info(f"Server address: {url_list}")
            break
        except Exception as e:
            logging.warning(f"Error {e}; No server address, retrying ...")
            sleep(30)

# tokenizer = AutoTokenizer.from_pretrained(args.model)
system_prompt = "You are a helpful assistant."
conv = [{"role": "system", "content": system_prompt}]

print(args.mode)

step_label = False
final_label = False
if '_step_label_' in args.model:
    step_label = True
    if args.mode == 'critic':
        args.best_of = 1
        args.num_shot = 1
        args.temperature = 0.0
elif '_final_label_' in args.model:
    final_label = True
    if args.mode == 'critic':
        args.best_of = 1
        args.temperature = 0.0


def infer_api(url, prompt, num_return_sequences=1, generation_prefix='', max_send_num=16, best_of=None, stop=["<|eot_id|>", "[END OF SOLUTION]"]):
    url = random.choice(url_list)
    # Build the prompt with a conversation template
    msg = prompt
    new_conv = copy.deepcopy(conv)
    new_conv.append({"role": "user", "content": msg})
    max_send_num = min(max_send_num, num_return_sequences)
    remaining_num = num_return_sequences

    payload = {
        "model": args.model,
        "messages": new_conv,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    total_output_list = []

    # Round up
    for _ in range((num_return_sequences + max_send_num - 1) // max_send_num):
        this_time_num = min(remaining_num, max_send_num)
        payload["n"] = this_time_num
        if best_of is not None:
            payload["best_of"] = max(this_time_num, best_of)

        for i in range(5):
            output_list = []
            try:
                payload["n"] = this_time_num
                response = requests.post(url, json=payload, timeout=240)
                logging.debug(response.text)
                response_load = json.loads(response.text)['choices']
                output_list = [item['message']['content'] for item in response_load]
                break
            except Exception as e:
                logging.error(e)
                sleep(0.1)
                logging.error('retry')
                pass
        total_output_list += output_list
        remaining_num -= this_time_num
    return total_output_list

def infer_instance_solve(url, instance, only_first_half=False):
    prompt = f'''## Python code problem\n{instance["prompt"]}\n\n-----\nBefore writing the code, think step by step, marking each step as "Step [i]:".\nYour final function {instance["entry_point"]} should be in the form ```python\\n[code]\\n```, at the end of your response.'''


    pre_generated_steps_list = []
    full_outputs = []
    correct_list = []
    for max_try in range(10): 
        outputs = infer_api(url, prompt, args.num_return_sequences, generation_prefix='', stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", ])
        if only_first_half and len_outputs > 1:
            outputs = outputs[:len_outputs//2]
        break
    
    references = [instance["test_list"]] * len(outputs)
    correct_list, pre_generated_steps_list, details = grade_answer(outputs, references)

    instance["pre_generated_steps"] = pre_generated_steps_list
    instance["full_outputs"] = outputs
    instance["correct"] = correct_list

    # logging.info(instance)

    return instance


def infer_instance_critic(url, instance, num_return_sequences=1, early_stop=False, best_of=8):
    problem = instance['prompt']
    pre_generated_steps = instance['pre_generated_steps'] 
    if len(instance['pre_generated_steps']) > 0 and type(instance['pre_generated_steps'][0]) is not list:
        pre_generated_steps = [pre_generated_steps]
    critics = []
    pred_labels_list = []
    refinements = []
    refine_correct_list = []
    avg_pred_score_list = []

    for i, steps_completion in enumerate(pre_generated_steps):

        start_idx = 1 if steps_completion[0].startswith("def") else 3
        line_list = steps_completion[:start_idx] + [f"Line {i+1}: {step}" for i, step in enumerate(steps_completion[start_idx:])]
        line_str = '\n'.join(line_list)
        original_line_str = '\n'.join(steps_completion)
        steps_str = instance['full_outputs'][i].replace(original_line_str, line_str)
        logging.info("Attempt:\n" + steps_str)
        len_steps = len(re.findall(r"Step \d+", steps_str))
        len_lines = len(line_list) - start_idx

        if step_label:
            prompt = f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n For each line, you only need to output "Line [i] is correct" if right, or "Line [i] is incorrect" if wrong. Do not provide anything else in your output.'''
        elif final_label:
            prompt = f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n For the whole attempt, you only need to output "Each step from Line 1 to Line {len_steps} is correct." if right, or "Some line from Line 1 to Line {len_steps} is incorrect" if wrong. Do not provide anything else in your output.'''
        else:
            prompt =  f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step and line by line.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Step [i] is correct" or "Step [i] is incorrect".\n - At the end of the evaluation for each line, YOU MUST articulate the conclusion using the format "Line [i] is correct" or "Line [i] is incorrect".'''

        this_critics = infer_api(url, prompt, num_return_sequences=num_return_sequences, best_of=best_of, stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", "<correction>"], generation_prefix='Line 1')
        pred_labels = []
        pred_scores = []
        for critic in this_critics:
            logging.info(critic)
            this_pred_labels_1 = parse_label(critic, len_steps, step="Step")
            this_pred_labels_2 = parse_label(critic, len_lines, step="Line")
            this_pred_labels = this_pred_labels_1 + this_pred_labels_2
            pred_labels.append(this_pred_labels)
            pred_scores.append(1 if -1 not in this_pred_labels and 0 not in this_pred_labels else 0)
        
        this_correct = instance['correct'][i] if type(instance['correct']) is list else instance['correct']
        avg_pred_score = sum(pred_scores) / len(pred_scores) if pred_scores != [] else 0
        avg_pred_score_list.append(avg_pred_score)
        logging.info(this_correct)
        logging.info(pred_labels)
        logging.info(avg_pred_score)

        refinement = '' # parse_correction_and_changed_answer(this_critics[0])
        refine_correct = None

        critics.append(this_critics[0] if len(this_critics) == 1 else this_critics)
        pred_labels_list.append(pred_labels[0] if len(pred_labels) == 1 else pred_labels)
        refinements.append(refinement)
        refine_correct_list.append(refine_correct)

        if early_stop and avg_pred_score == 1:
            break

    new_instance = copy.deepcopy(instance)
    new_instance["pred_labels"] = copy.deepcopy(pred_labels_list)
    new_instance['avg_pred_score'] = copy.deepcopy(avg_pred_score_list)
    # Select the index of max avg_pred_score
    max_avg_pred_score = max(avg_pred_score_list)
    max_avg_pred_score_idx = avg_pred_score_list.index(max_avg_pred_score)
    best_of_n_correct = pred_labels_list[max_avg_pred_score_idx]
    new_instance['bon_score'] = max_avg_pred_score
    new_instance['bon_idx'] = max_avg_pred_score_idx
    new_instance['bon_correct'] = copy.deepcopy(best_of_n_correct)
    logging.info(f"Best of {len(avg_pred_score_list)}: ={best_of_n_correct}")
    new_instance['critics'] = copy.deepcopy(critics)

    return new_instance

def infer_instance_refine(url, instance, best_of=8):
    problem = instance['prompt']
    full_outputs = []
    refinements = []
    refine_correct_list = []
    wrong_step_no_list = []
    if len(instance['pre_generated_steps']) > 0 and type(instance['pre_generated_steps'][0]) is not list:
        instance['pre_generated_steps'] = [instance['pre_generated_steps']]

    for steps_completion, full_output, step_wise_critc in zip(instance['pre_generated_steps'], instance['full_outputs'], instance['critics']):
        start_idx = 1 if steps_completion[0].startswith("def") else 3
        line_list = steps_completion[:start_idx] + [f"Line {i+1}: {step}" for i, step in enumerate(steps_completion[start_idx:])]
        line_str = '\n'.join(line_list)
        original_line_str = '\n'.join(steps_completion)
        steps_str = full_output.replace(original_line_str, line_str)
        # logging.info(steps_str)
        len_steps = len(re.findall(r"Step \d+", steps_str))
        len_lines = len(line_list) - start_idx

        try:
            if step_label:
                wrong_step_criticism = re.findall(r"Line \d+ is incorrect", step_wise_critc)[0]
                wrong_step_no = re.findall(r"Line (\d+) is incorrect", wrong_step_criticism)[0]
            elif final_label:
                wrong_step_criticism = re.findall(r"Some line from Line 1 to Line \d+ is incorrect", step_wise_critc)[0]
                wrong_step_no = 1
            else:
                wrong_step_no = re.findall(r"Line (\d+) is incorrect", step_wise_critc)[0]
                wrong_step_criticism = re.findall(f"Line {wrong_step_no}" + r".*Line \d+ is incorrect", step_wise_critc, re.DOTALL)[0]
        except Exception as e:
            try:
                wrong_step_no = re.findall(r"Step (\d+) is incorrect", step_wise_critc)[0]
                wrong_step_criticism = re.findall(f"Step {wrong_step_no}" + r".*Step \d+ is incorrect", step_wise_critc, re.DOTALL)[0]
            except Exception as e:
                logging.error(e)
                refinements.append('')
                full_outputs.append(full_output)
                refine_correct_list.append(None)
                wrong_step_no_list.append(-1)
                continue

        prompt =  f'''How do you refine the following attempt with respect to the problem, given the criticism? You shall write another complete Python function, in the format ```python\\n[code]\\n```.\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n<criticism>\n{wrong_step_criticism}\n</criticism>'''

        logging.debug(prompt)
        generation_prefix = '<correction>\nLine' if not final_label else 'Line'
        output = infer_api(url, prompt, generation_prefix=generation_prefix, best_of=best_of, stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", ])[0]
        logging.info(output)
        correct_list, pre_generated_steps_list, details = grade_answer([output], [instance["test_list"]])
        refinement = pre_generated_steps_list[0]
        logging.info(refinement)
        refine_correct = correct_list[0]
        logging.info(refine_correct)

        full_outputs.append(output)
        refinements.append(refinement)
        refine_correct_list.append(refine_correct)
        wrong_step_no_list.append(wrong_step_no)


    instance['full_outputs'] = full_outputs
    instance['refinements'] = refinements
    instance['refine_correct'] = refine_correct_list
    instance['wrong_step_no'] = wrong_step_no_list
    return instance

def infer_instance_iterative_refine(url, instance, max_refine_depth=args.max_refine_depth, max_refine_restart=args.max_refine_restart):
    previous_refine_record = []
    refine_instance = instance
    if type(instance['pre_generated_steps'][0]) is not list:
        instance['pre_generated_steps'] = [instance['pre_generated_steps']]
    else:
        # Only select 1 sample
        instance['pre_generated_steps'] = instance['pre_generated_steps'][:1]
        instance['pred_labels'] = instance['pred_labels'][:1]
        instance['correct'] = instance['correct'][:1]
        instance['critics'] = instance['critics'][:1]
        instance['avg_pred_score'] = instance['avg_pred_score'][:1]
        del instance['bon_score']
        del instance['bon_idx']
    
    if type(instance['correct']) is not list:
        instance['correct'] = [instance['correct']]

    all_predict_true = False if any(-1 in pred_labels for pred_labels in instance['pred_labels']) else True
    if all_predict_true:
        logging.info(f"No need for refinement. Actual correct: {instance['correct']}")
        return instance

    for restart in range(1, max_refine_restart+1, 1):
        previous_refine_record.append([])
        input_instance = copy.deepcopy(instance)
        for round in range(1, max_refine_depth+1, 1):
            refine_instance = infer_instance_refine(url, input_instance)

            previous_refine_record[-1].append(
                {
                    'round': round,
                    'pre_generated_steps': copy.deepcopy(refine_instance['pre_generated_steps']),
                    'critics': copy.deepcopy(refine_instance['critics']),
                    'wrong_step_no': copy.deepcopy(refine_instance['wrong_step_no']),
                    'refinements': copy.deepcopy(refine_instance['refinements']),
                    'refine_correct': copy.deepcopy(refine_instance['refine_correct']),
                }
            )

            # Replace the pre_generated_steps with the refined steps
            for i, refinement in enumerate(refine_instance['refinements']):
                if refinement != '':
                    refine_instance['pre_generated_steps'][i] = refinement

            input_instance = copy.deepcopy(refine_instance)
            input_instance = infer_instance_critic(url, input_instance, best_of=args.best_of)
            logging.info(f"Round {round}: {input_instance['pred_labels']}")
            all_predict_true = False if any(-1 in pred_labels for pred_labels in input_instance['pred_labels']) else True
            if all_predict_true:
                refine_instance["critics"] = input_instance["critics"]
                refine_instance["pred_labels"] = input_instance["pred_labels"]
                logging.info(f"Exit at restart {restart}, round {round}, refine_correct: {refine_instance['refine_correct']}")
                refine_instance['previous_refine_record'] = previous_refine_record
                return refine_instance
            
        logging.info(f"Restart {restart} failed")
    
    refine_instance['previous_refine_record'] = previous_refine_record
    return refine_instance

def infer_instance_reject_sampling(url, instance, max_retry_cnt=64, num_critic=1):
    # First, infer the instance
    # Retry until the critic is correct, or the max_retry_cnt is reached
    instance = copy.deepcopy(instance)
    for retry_cnt in range(max_retry_cnt):
        solution_instance = copy.deepcopy(infer_instance_solve(url, instance, only_first_half=False))

        critic_instance = copy.deepcopy(infer_instance_critic(url, solution_instance, num_return_sequences=num_critic, early_stop=True))
        if critic_instance['bon_score'] == 1:
            for key in ['pre_generated_steps', 'correct', 'critics', 'pred_labels']:
                try:
                    critic_instance[key] = [critic_instance[key][critic_instance['bon_idx']]] + critic_instance[key][:critic_instance['bon_idx']] + critic_instance[key][critic_instance['bon_idx']+1:]
                except Exception as e:
                    logging.error(e)
                    logging.error(f"Error in reranking {key}")
                    logging.error(critic_instance)
                    input()
                    raise
            critic_instance['retry_cnt'] = retry_cnt
            logging.info("Success")
            return critic_instance
        elif retry_cnt == max_retry_cnt - 1:
            # Failed
            critic_instance['retry_cnt'] = max_retry_cnt
            return critic_instance
        logging.info(f"Retry {retry_cnt}")

def infer_instance(url, instance):
    if args.mode == 'critic':
        return infer_instance_critic(url, instance, best_of=args.best_of)
    elif args.mode == 'refine':
        return infer_instance_refine(url, instance, best_of=args.best_of)
    elif args.mode == 'iterative_refine':
        return infer_instance_iterative_refine(url, instance)
    elif args.mode == 'solve':
        return infer_instance_solve(url, instance)
    elif args.mode == 'reject_sampling':
        return infer_instance_reject_sampling(url, instance, max_retry_cnt=args.max_retry_cnt)

def infer_file(port, instance_list, write_path, request_batch_size):
    fw = open(write_path, 'a')
    url = f"http://localhost:{port}/generate"

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(request_batch_size) as p:
            for new_instance in tqdm(p.map(lambda x: infer_instance(url, x), instance_list), total=len(instance_list)):
                print(json.dumps(new_instance), file=fw)
                fw.flush()

    fw.close()

if __name__=='__main__':
    with open(args.src, 'r') as f:
        instance_list = f.readlines()[:args.max_instance]
    instance_list = [json.loads(instance) for instance in instance_list]

    try:
        with open(args.tgt, 'r') as f:
            previous_instance_list = f.readlines()
        previous_instance_list = [json.loads(instance) for instance in previous_instance_list]
        previous_problem_set = set([instance['prompt'] for instance in previous_instance_list])
        instance_list = [instance for instance in instance_list if instance['prompt'] not in previous_problem_set]
    except:
        pass

    infer_file(args.port, instance_list, args.tgt, args.request_batch_size)
