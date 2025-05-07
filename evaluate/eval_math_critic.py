import json
from sklearn.metrics import precision_recall_fscore_support
import argparse
import numpy as np
import sys 
import collections
import logging
import math

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

sys.path.append("..") 
sys.path.append(".") 

from grading.grader import grade_answer

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
parser.add_argument('--max_instance', type=int, default=66666)
parser.add_argument('--max_response', type=int, default=512)
args = parser.parse_args()

with open(args.pred_file, 'r') as f:
    data = f.readlines()

data = [json.loads(d) for d in data[:args.max_instance]]

pred_label_list = []
majority_vote_pred_label_list = []
gold_label_list = []

cnt = len(data)
correct_cnt = 0
no_problem_when_correct = 0
wrong_cnt = 0
found_error_when_wrong_cnt = 0

refine_unnecessary_cnt = 0.1
refine_unnecessary_but_unchanged_cnt = 0
refine_required_cnt = 0.1
refine_required_and_success_cnt = 0

correct_after_refinement_cnt = 0

majority_vote_correct = 0
majority_vote_after_critic_correct = 0
majority_vote_after_refine_correct = 0
pass_1_at_n_correct = 0

majority_vote_correct_after_refinement_cnt = 0
majority_vote_refine_required_cnt = 0
majority_vote_refine_required_and_success_cnt = 0
contain_correct_cnt = 0

problem_set = set([d['problem'] for d in data])
logging.debug(f"len_problem_set: {len(problem_set)}")

for instance in data:
    pred_labels = instance['pred_labels']
    instance['predict_answer'] = [a.replace(' ', '') for a in instance['predict_answer']]
    if pred_labels == []:
        logging.error("empty pred labels")
        continue

    
    for this_pred_labels, this_correct in zip(instance['pred_labels'][:1], instance['correct'][:1]):
        if this_pred_labels == []:
            continue
        if type(this_pred_labels[0]) is list:
            this_pred_labels = this_pred_labels[0]
        critic_pred = 1 if -1 not in this_pred_labels else -1
        pred_label_list.append(critic_pred)
        this_correct = 1 if this_correct else -1
        gold_label_list.append(this_correct)
        
    
            
    if type(instance['correct']) is not list:
        instance['correct'] = [instance['correct']]
    correct = instance['correct'][0]
    # logging.debug(f"pred_label: {pred_label}, correct: {correct}")
    # input()
    pass_1_at_n_correct += 1 if True in instance['correct'] else 0
    refine_correct = instance['refine_correct'][0] if instance['refine_correct'] != [] and instance['refinements'][0] != '' else None
    correct_after_refinement_cnt += refine_correct if refine_correct is not None else (correct == 1)

    gold_answer = instance['answer']

    # Majority vote
    # Randomly select indices with length of `max_response`
    random_index = np.random.choice(len(instance['refine_answers']), min(args.max_response, len(instance['refine_answers'])), replace=False)
    refine_answers = [instance['refine_answers'][i] for i in random_index]
    original_predict_answer = [instance['predict_answer'][i] for i in random_index]
    original_predict_answer = [a.strip() for a in original_predict_answer]
    all_pred_labels = [instance['pred_labels'][i] for i in random_index]
    refine_answers_replaced = [oa if a == '' else a for oa, a in zip(original_predict_answer, refine_answers)]
    refine_answers_changed = [a for oa, a in zip(original_predict_answer, refine_answers) if a not in ['', oa]]

    # Take the most frequent answer from original_predict_answer
    count_original_predict_answer = collections.Counter(original_predict_answer)
    most_frequent_original_predict_answer = count_original_predict_answer.most_common(1)[0][0] if len(original_predict_answer) > 0 else ""
    this_majority_vote_correct = grade_answer(gold_answer, most_frequent_original_predict_answer)
    majority_vote_correct += this_majority_vote_correct
    logging.debug(f"original_predict_answer:\n {count_original_predict_answer}")

    # Filter original_predict_answer using the corresponding instance['pred_labels']
    critic_filtered_predict_answer = [a.strip() for a, l in zip(original_predict_answer, all_pred_labels) if -1 not in l]

    count_critic_filtered_predict_answer = collections.Counter(critic_filtered_predict_answer)
    most_frequent_critic_filtered_predict_answer = count_critic_filtered_predict_answer.most_common(1)[0][0] if len(critic_filtered_predict_answer) > 0 else ""

    # Fall back, since most predictions are filtered out
    if (len(set(critic_filtered_predict_answer)) > 1 and len(critic_filtered_predict_answer) < min(6, len(original_predict_answer) // 16)) or most_frequent_critic_filtered_predict_answer == "":
        most_frequent_critic_filtered_predict_answer = most_frequent_original_predict_answer
        logging.debug(f"Majority vote fallback")
    logging.debug(f"most_frequent_critic_filtered_predict_answer:\n {count_critic_filtered_predict_answer}")
    this_majority_vote_after_critic_correct = grade_answer(gold_answer, most_frequent_critic_filtered_predict_answer)
    majority_vote_after_critic_correct += this_majority_vote_after_critic_correct
    # print(grade_answer(gold_answer, most_frequent_critic_filtered_predict_answer), "gold", gold_answer, "most", most_frequent_critic_filtered_predict_answer, '\n')
    if this_majority_vote_correct and not this_majority_vote_after_critic_correct:
        logging.debug("Critic filtered prediction is worse than original prediction")
    elif not this_majority_vote_correct and this_majority_vote_after_critic_correct:
        logging.debug("Critic filtered prediction is better than original prediction")
    # Check if the correct answer is in the critic filtered answers
    if this_majority_vote_after_critic_correct or gold_answer in critic_filtered_predict_answer:
        logging.debug(f"Contain correct answer")
    else:
        logging.debug(f"Miss correct answer")
    logging.debug(f"{grade_answer(gold_answer, most_frequent_critic_filtered_predict_answer)}, gold, {gold_answer}, most, {most_frequent_critic_filtered_predict_answer}\n\n")

    most_frequent_refine_answer = collections.Counter(refine_answers_replaced).most_common(1)[0][0] if len(refine_answers_replaced) > 0 else ""
    majority_vote_refine_correct = grade_answer(gold_answer, most_frequent_refine_answer)
    # if not majority_vote_refine_correct:
    #     print("gold", gold_answer,  "most", most_frequent_refine_answer, '\n')
    majority_vote_after_refine_correct += majority_vote_refine_correct

    # Take the most frequent answer using collection
    most_frequent_answer = collections.Counter(refine_answers_changed).most_common(1)[0][0] if len(refine_answers_changed) > 0 else ""

    # If refine is more frequent than not refine, then we take the majority as taking the refine
    not_refine_list = [a for a in refine_answers if a == '']
    if len(refine_answers_changed) <= len(refine_answers) // 2:
        # No refine
        # majority_vote_predict_label = 1
        majority_vote_refine_correct = correct
        majority_vote_correct_after_refinement_cnt += correct
    else:
        # Refine
        # majority_vote_predict_label = -1
        majority_vote_refine_correct = grade_answer(gold_answer, most_frequent_answer)
        # print(majority_vote_refine_correct, "gold", gold_answer, "original", original_predict_answer,  "most", most_frequent_answer, refine_answers, '\n')
        majority_vote_correct_after_refinement_cnt += majority_vote_refine_correct

    all_pred_labels = [-1 if -1 in p else 1 for p in instance['pred_labels']]
    majority_vote_predict_label = -1 if all_pred_labels.count(-1) > all_pred_labels.count(1) else 1
    majority_vote_pred_label_list.append(majority_vote_predict_label)

    # gold answer cotained in the refinement answers
    for a in refine_answers_replaced:
        if grade_answer(gold_answer, a):
            contain_correct_cnt += 1
            break
    

    if correct == 1:
        correct_cnt += 1
        if -1 not in pred_labels:
            no_problem_when_correct += 1
        if refine_correct is not None:
            refine_unnecessary_cnt += 1
            refine_unnecessary_but_unchanged_cnt += refine_correct
    else:
        wrong_cnt += 1
        if -1 in pred_labels:
            found_error_when_wrong_cnt += 1
        if refine_correct is not None:
            refine_required_cnt += 1
            refine_required_and_success_cnt += refine_correct
        if majority_vote_predict_label == -1:
            majority_vote_refine_required_cnt += 1
            majority_vote_refine_required_and_success_cnt += majority_vote_refine_correct

gold_label_list = np.array(gold_label_list)
pred_label_list = np.array(pred_label_list)
majority_vote_pred_label_list = np.array(majority_vote_pred_label_list)
presicion, recall, f1, support = precision_recall_fscore_support(gold_label_list, pred_label_list, labels=[1, -1], pos_label=-1, average='binary', zero_division=0)
logging.debug(f"set(pred_label_list): {set(pred_label_list)}, set(gold_label_list): {set(gold_label_list)}")
accuracy = (gold_label_list == pred_label_list).sum() / len(gold_label_list)
correct_solution_refine_unchanged_rate = 100.0 * refine_unnecessary_but_unchanged_cnt / refine_unnecessary_cnt
wrong_solution_refine_success_rate = 100.0 * refine_required_and_success_cnt / refine_required_cnt
initial_solution_accuracy = 100.0 * correct_cnt / cnt
after_refinement_solution_accuracy = 100.0 * correct_after_refinement_cnt / cnt
majority_vote_accuracy = 100.0 * majority_vote_correct / cnt
majority_vote_after_critics_accuracy = 100.0 * majority_vote_after_critic_correct / cnt
majority_vote_after_refine_solution_accuracy = 100.0 * majority_vote_after_refine_correct / cnt
pass_1_at_n_accuracy = 100.0 * pass_1_at_n_correct / cnt


print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (presicion*100, recall*100, f1*100, accuracy * 100, correct_solution_refine_unchanged_rate, wrong_solution_refine_success_rate, initial_solution_accuracy, after_refinement_solution_accuracy, pass_1_at_n_accuracy, majority_vote_accuracy, majority_vote_after_critics_accuracy, majority_vote_after_refine_solution_accuracy), end='\t')
print()
