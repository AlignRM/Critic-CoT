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

import logging

logging.basicConfig(level=logging.INFO)

sys.path.append("..") 
sys.path.append(".") 

from grading.grader import grade_answer
from utils import parse_label, parse_correction_and_changed_answer, extract_boxed_expressions_custom

from transformers import AutoTokenizer
import copy

random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--num_return_sequences', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--top_k', type=int, default=80)
parser.add_argument('--max_tokens', type=int, default=1024)
parser.add_argument('--max_instance', type=int, default=10000000)
parser.add_argument('--token', type=str, default='')
parser.add_argument('--mode', type=str, default='critic')
parser.add_argument('--repetition_penalty', type=float, default=1.1)

parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--address_file", type=str, default="server/address.txt")
parser.add_argument("--request_batch_size", type=int, default=32)

args = parser.parse_args()

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

print(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
system_prompt = "You are a helpful assistant."
conv = [{"role": "system", "content": system_prompt}]

def infer_api(url, prompt, num_return_sequences=1, generation_prefix='', max_send_num=64, best_of=None, stop=["<|eot_id|>", "[END OF SOLUTION]"]):
    url = random.choice(url_list)
    # Build the prompt with a conversation template
    msg = prompt
    new_conv = copy.deepcopy(conv)
    new_conv.append({"role": "user", "content": msg})
    final_prompt = tokenizer.apply_chat_template(new_conv, tokenize=False, add_generation_prompt=True)
    max_send_num = min(max_send_num, num_return_sequences)
    remaining_num = num_return_sequences

    payload = {
        "prompt": final_prompt + generation_prefix,
        "use_beam_search": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "stop": stop, # LLAMA-3
        "max_tokens": args.max_tokens,
        # "n": min(num_return_sequences, max_send_num),
    }
    len_final_prompt = len(final_prompt) # generation_prefix would be kept in the output
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
                response_load = json.loads(response.text)['text']
                output_list = [item[len_final_prompt:] for item in response_load]
                break
            except Exception as e:
                logging.error(e)
                sleep(0.1)
                logging.error('retry')
                pass
        total_output_list += output_list
        remaining_num -= this_time_num
    return total_output_list

def infer_instance_solve(url, instance, num_return_sequences=10, only_first_half=False):
    prompt = f'''## Math problem\n{instance["problem"]}\n\n-----\n''' + '''Solve the problem step by step, marking each step as "Step [i]:".\nYour final answer should be in the form \\boxed{answer}, at the end of your response.'''

    pre_generated_steps_list = []
    predict_answers = []
    correct_list = []
    for max_try in range(10): 
        outputs = infer_api(url, prompt, num_return_sequences=num_return_sequences, generation_prefix='Step 1', stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", ])
        len_outputs = len(outputs)
        if only_first_half and len_outputs > 1:
            outputs = outputs[:len_outputs//2]

        for output in outputs:
            step_split = []
            predict_answer = ''
            correct = False
            try:
                step_split = re.split("\n*\**Step [0-9]+\**: ?", output)[1:]
                predict_answer_latex_str = re.findall(r'(\\boxed\{.*\})', output, re.DOTALL)[-1]
                predict_answer = extract_boxed_expressions_custom(predict_answer_latex_str)

                correct = grade_answer(predict_answer, instance["answer"])
                assert step_split != []
            except Exception as e:
                logging.error(e)
                logging.error(f"Error in the output solution: {output}")
                continue

            pre_generated_steps_list.append(step_split)
            predict_answers.append(predict_answer)
            correct_list.append(correct)
        if correct_list != []:
            break
    if correct_list == []:
        correct_list = [False]
        pre_generated_steps_list = [["Empty"]]
        predict_answers = [""]

    instance["pre_generated_steps"] = pre_generated_steps_list
    instance["predict_answer"] = predict_answers
    instance["correct"] = correct_list

    # logging.info(instance)

    return instance


def infer_instance_critic(url, instance, num_return_sequences=1, early_stop=False, best_of=1):
    problem = instance['problem']
    gold_answer = instance["answer"]
    predict_answers = instance['predict_answer']
    pre_generated_steps = instance['pre_generated_steps'] 
    if len(instance['pre_generated_steps']) > 0 and type(instance['pre_generated_steps'][0]) is not list:
        pre_generated_steps = [pre_generated_steps]
        predict_answers = [predict_answers]
    critics = []
    pred_labels_list = []
    refinements = []
    refine_answers = []
    refine_correct_list = []
    avg_pred_score_list = []

    for i, steps_completion in enumerate(pre_generated_steps):
        steps_list = [f"Step {i+1}: {step}" for i, step in enumerate(steps_completion)]
        steps_str = '\n\n'.join(steps_list)
        len_steps = len(steps_list)

        prompt =  f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Step [i] is correct" or "Step [i] is incorrect".'''

        this_critics = infer_api(url, prompt, num_return_sequences=num_return_sequences, best_of=best_of, stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", "<correction>"], generation_prefix='Step 1')
        pred_labels = []
        pred_scores = []
        for critic in this_critics:
            logging.info(critic)
            this_pred_labels = parse_label(critic, len(steps_list))
            pred_labels.append(this_pred_labels)
            pred_scores.append(1 if -1 not in this_pred_labels and 0 not in this_pred_labels else 0)
        
        this_correct = instance['correct'][i] if type(instance['correct']) is list else instance['correct']
        avg_pred_score = sum(pred_scores) / len(pred_scores) if pred_scores != [] else 0
        avg_pred_score_list.append(avg_pred_score)
        logging.info(this_correct)
        logging.info(pred_labels)
        logging.info(avg_pred_score)

        refinement, refine_answer = '', '' # parse_correction_and_changed_answer(this_critics[0])
        refine_correct = None

        critics.append(this_critics[0] if len(this_critics) == 1 else this_critics)
        pred_labels_list.append(pred_labels[0] if len(pred_labels) == 1 else pred_labels)
        refinements.append(refinement)
        refine_answers.append(refine_answer)
        refine_correct_list.append(refine_correct)

        if early_stop and avg_pred_score == 1:
            break

    new_instance = copy.deepcopy(instance)
    new_instance["pred_labels"] = copy.deepcopy(pred_labels_list)
    new_instance['avg_pred_score'] = copy.deepcopy(avg_pred_score_list)
    # Select the index of max avg_pred_score
    max_avg_pred_score = max(avg_pred_score_list)
    max_avg_pred_score_idx = avg_pred_score_list.index(max_avg_pred_score)
    best_of_n_answer = predict_answers[max_avg_pred_score_idx]
    best_of_n_correct = grade_answer(best_of_n_answer, gold_answer)
    new_instance['bon_score'] = max_avg_pred_score
    new_instance['bon_idx'] = max_avg_pred_score_idx
    new_instance['bon_answer'] = copy.deepcopy(best_of_n_answer)
    new_instance['bon_correct'] = copy.deepcopy(best_of_n_correct)
    logging.info(f"Best of {len(avg_pred_score_list)}: {best_of_n_answer}, {best_of_n_correct}")
    new_instance['refine_correct'] = copy.deepcopy(refine_correct_list)
    new_instance['refine_answers'] = copy.deepcopy(refine_answers)
    new_instance['critics'] = copy.deepcopy(critics)
    new_instance['refinements'] = copy.deepcopy(refinements)

    return new_instance

def infer_instance_refine(url, instance, best_of=1, num_return_sequences=1):
    problem = instance['problem']
    gold_answer = instance["answer"]
    refinements = []
    refine_answers = []
    refine_correct_list = []
    wrong_step_no_list = []
    if len(instance['pre_generated_steps']) > 0 and type(instance['pre_generated_steps'][0]) is not list:
        instance['pre_generated_steps'] = [instance['pre_generated_steps']]

    for steps_completion, step_wise_critc in zip(instance['pre_generated_steps'], instance['critics']):
        steps_list = [f"Step {i+1}: {step}" for i, step in enumerate(steps_completion)]
        steps_str = '\n\n'.join(steps_list)

        try:
            wrong_step_no = re.findall(r"Step (\d+) is incorrect", step_wise_critc)[0]
            assert int(wrong_step_no) <= len(steps_list)
            wrong_step_criticism = re.findall(f"Step {wrong_step_no}" + r".*Step \d+ is incorrect", step_wise_critc, re.DOTALL)[0]
        except Exception as e:
            # logging.error(e)
            refinements.append('')
            refine_answers.append('')
            refine_correct_list.append(None)
            wrong_step_no_list.append(-1)
            continue

        prompt =  f'''How do you refine the following attempt with respect to the problem, given the criticism?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n<criticism>\n{wrong_step_criticism}\n</criticism>'''

        logging.debug(prompt)
        generation_prefix = '<correction>\nStep'
        output = infer_api(url, prompt, num_return_sequences=num_return_sequences, generation_prefix=generation_prefix, best_of=best_of, stop=["<|eot_id|>", "[END OF SOLUTION]", "[END OF CRITIC]", ])[0]
        logging.info(output)
        refinement, refine_answer = parse_correction_and_changed_answer(output)
        refine_correct = grade_answer(refine_answer, gold_answer) if refinement != '' else None
        refinements.append(refinement)
        refine_answers.append(refine_answer)
        refine_correct_list.append(refine_correct)
        wrong_step_no_list.append(wrong_step_no)


    instance['refinements'] = refinements
    instance['refine_answers'] = refine_answers
    instance['refine_correct'] = refine_correct_list
    instance['wrong_step_no'] = wrong_step_no_list
    return instance

def infer_instance(url, instance, max_repeat=16, num_return_sequences=10, critic_refine_retry=16, **args):
    # First, sample solutions, with correct and incorrect cases all equal to num_return_sequences // 2 
    output_dict = {
        True: [],
        False: [],
    }
    final_answer_dict = {
        True: [],
        False: [],
    }
    half_num_seq = num_return_sequences // 2

    for _ in range(max_repeat):
        new_instance = infer_instance_solve(url=url, instance=instance, num_return_sequences=num_return_sequences*2)
        pre_generated_steps_list = new_instance['pre_generated_steps']
        correct_list = new_instance['correct']
        predict_answer_list = new_instance['predict_answer']

        logging.info(f"Predict Answer: {predict_answer_list}")
        logging.info(f"Gold Answer: {new_instance['answer']}")

        for correct, pre_generated_steps, predict_answer in zip(correct_list, pre_generated_steps_list, predict_answer_list):
            # Temporarily
            # if correct:
            #     continue
            output_dict[correct].append(pre_generated_steps)
            final_answer_dict[correct].append(predict_answer)

        logging.info(f"Generation solution, Correct: {len(output_dict[True])}, Incorrect: {len(output_dict[False])}")

        if len(output_dict[True]) >= half_num_seq and len(output_dict[False]) >= half_num_seq:
            logging.info("Generation solution finished")
            break

    # Truncate the number of solutions
    output_dict[True] = output_dict[True][:num_return_sequences]
    output_dict[False] = output_dict[False][:num_return_sequences]

    # # Only sample predictions
    # instance['pre_generated_steps'] = output_dict[True][:half_num_seq] + output_dict[False][:half_num_seq]
    # instance['predict_answer'] = final_answer_dict[True][:half_num_seq] + final_answer_dict[False][:half_num_seq]
    # instance['correct'] = [True] * len(output_dict[True][:half_num_seq]) + [False] * len(output_dict[False][:half_num_seq])
    # instance['pred_labels'] = []
    # instance['refinements'] = []
    # instance['refine_answers'] = []
    # instance['refine_correct'] = []
    # instance['failed_cases'] = []

    # return instance


    # Then, sample critics
    pre_generated_steps_list = []
    predict_answers = []
    correct_list = []

    critics = []
    pred_labels = []
    refinements = []
    refine_answers = []
    refine_correct_list = []

    # Aslo save failed cases
    failed_cases = {
        "pre_generated_steps": [],
        "predict_answer": [],
        "correct": [],
    }

    # For correct solutions, the critic for each step is always correct
    for i in range(len(output_dict[True])):
        instance['pre_generated_steps'] = [output_dict[True][i]]
        instance['predict_answer'] = [final_answer_dict[True][i]]

        # Retry at most max_repeat times
        critic_success = False
        for j in range(max_repeat):
            critic_instance = infer_instance_critic(url, instance)
            critic_list = critic_instance['critics']
            pred_labels_list = critic_instance['pred_labels']

            # Check if the critic is correct
            correct = False if -1 in pred_labels_list[0] or 0 in pred_labels_list[0] else True

            # If the critic is correct, then accept; otherwise, reject
            if not correct:
                continue
            pre_generated_steps_list.append(output_dict[True][i])
            predict_answers.append(final_answer_dict[True][i])
            correct_list.append(True)
            critics.append(critic_list[0])
            pred_labels.append(pred_labels_list[0])
            logging.info(f"Add correct critic {pred_labels_list[0]} ")
            refinements.append("")
            refine_answers.append("")
            refine_correct_list.append(None)
            critic_success = True
            break

        if not critic_success and len(failed_cases["pre_generated_steps"]) < num_return_sequences:
            failed_cases["pre_generated_steps"].append(output_dict[True][i])
            failed_cases["predict_answer"].append(final_answer_dict[True][i])
            failed_cases["correct"].append(True)

        # If the collect number of correct solutions is enough, then break
        if len(pre_generated_steps_list) >= half_num_seq:
            # Empty the failed cases
            failed_cases["pre_generated_steps"] = []
            failed_cases["predict_answer"] = []
            failed_cases["correct"] = []
            break

    cnt_correct_solution_correct_critic = len(pre_generated_steps_list)
    logging.info(f"Collect critics for correct solutions, Cnt: {cnt_correct_solution_correct_critic}")

    # For incorrect solutions, the critic for some step is incorrect, and the refinement would reach the correct answer
    for i in range(len(output_dict[False])):
        instance['pre_generated_steps'] = [output_dict[False][i]]
        instance['predict_answer'] = [final_answer_dict[False][i]]

        # Retry at most `critic_refine_retry` times
        critic_success = False
        for j in range(critic_refine_retry):
            critic_instance = infer_instance_critic(url, instance)
            critic_list = critic_instance['critics']
            pred_labels_list = critic_instance['pred_labels']

            # Check if the critic detect the incorrect solution
            correct = False if -1 in pred_labels_list[0] else True

            # If the critic say it's correct, then reject; otherwise, refine
            if correct:
                continue

            refine_instance = infer_instance_refine(url, critic_instance)

            # Check if the refinement reach the correct answer; if not, reject
            refine_correct = refine_instance['refine_correct'][0]

            if refine_correct is None or not refine_correct:
                continue
            
            pre_generated_steps_list.append(output_dict[False][i])
            predict_answers.append(final_answer_dict[False][i])
            correct_list.append(False)
            critics.append(critic_list[0])
            pred_labels.append(pred_labels_list[0])
            logging.info(f"Add wrong critic {pred_labels_list[0]} ")

            refinements.append(refine_instance['refinements'][0])
            refine_answers.append(refine_instance['refine_answers'][0])
            refine_correct_list.append(refine_correct)
            critic_success = True
            break

        if not critic_success and len(failed_cases["pre_generated_steps"]) < 2 * num_return_sequences:
            failed_cases["pre_generated_steps"].append(output_dict[False][i])
            failed_cases["predict_answer"].append(final_answer_dict[False][i])
            failed_cases["correct"].append(False)

        # If the collect number of refine solutions is enough, then break
        if len(pre_generated_steps_list) >= num_return_sequences:
            break

    logging.info(f"Collect critics for incorrect solutions, Cnt: {len(pre_generated_steps_list) - cnt_correct_solution_correct_critic}")

    instance['pre_generated_steps'] = pre_generated_steps_list
    instance['predict_answer'] = predict_answers
    instance['correct'] = correct_list
    instance['critics'] = critics
    instance['pred_labels'] = pred_labels
    instance['refinements'] = refinements
    instance['refine_answers'] = refine_answers
    instance['refine_correct'] = refine_correct_list
    instance['failed_cases'] = failed_cases

    return instance

        


def infer_file(ip, port, instance_list, write_path, request_batch_size, **args):
    fw = open(write_path, 'a')
    url = f"http://{ip}:{port}/generate"

    with logging_redirect_tqdm():
        try:
            with ThreadPoolExecutor(request_batch_size) as p:
                for new_instance in tqdm(p.map(lambda instance: infer_instance(url, instance, **args), instance_list), total=len(instance_list)):
                    json.dump(new_instance, fw, ensure_ascii=False)
                    fw.write('\n')
                    fw.flush()
        except Exception as e:
            print(e)
            p.shutdown()

    fw.close()

if __name__=='__main__':
    with open(args.src, 'r') as f:
        instance_list = f.readlines()[:args.max_instance]
    instance_list = [json.loads(instance) for instance in instance_list]

    try:
        with open(args.tgt, 'r') as f:
            previous_instance_list = f.readlines()
        previous_instance_list = [json.loads(instance) for instance in previous_instance_list]
        previous_problem_set = set([instance['problem'] for instance in previous_instance_list])
        instance_list = [instance for instance in instance_list if instance['problem'] not in previous_problem_set]
    except:
        pass


    infer_file(instance_list=instance_list, write_path=args.tgt, **vars(args))
