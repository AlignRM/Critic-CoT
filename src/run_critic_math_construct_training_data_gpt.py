import openai

import argparse
from os import error
import re
from time import sleep
import random
from tqdm import tqdm
import ujson as json
from utils import parse_label, parse_correction_and_changed_answer

import sys 
sys.path.append("..") 
sys.path.append(".") 
from grading.grader import grade_answer

from multiprocessing import Pool

random.seed()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, )
parser.add_argument('--src', type=str, )
parser.add_argument('--tgt', type=str, )
parser.add_argument('--num_shot', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample', type=int, default=8)
parser.add_argument('--stop', type=str, default=None, nargs='+')
parser.add_argument('--max_try', type=int, default=64)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--disable_eos', action='store_true')
parser.add_argument('--max_tokens', type=int, default=3333)
parser.add_argument('--max_instance', type=int, default=999999)
parser.add_argument('--token', type=str, default='')
args = parser.parse_args()

messages = [
    {"role": "system", "content": "You are a helpful mathematician who has your own opinions"},
    {"role": "user", "content": '''How do you evaluate the following incorrect attempt?

<problem>
A Senate committee has 5 Democrats, 5 Republicans, and 1 Independent.  In how many ways can they sit around a circular table if all the members of each party all sit next to each other?  (Two seatings are considered equivalent if one is a rotation of the other.)
</problem>

<attempt>
Step 1: Identify the number of Democrats, Republicans, and Independents.

There are 5 Democrats, 5 Republicans, and 1 Independent.

Step 2: Arrange the members of each party together.

Since all members of each party must sit together, we can treat each party as a single entity.

So, we have 3 entities: Democrats, Republicans, and Independent.

Step 3: Arrange the entities around the circular table.

Since the table is circular, we consider arrangements as unique up to rotation. The number of arrangements of \( n \) distinct objects in a circle is \( (n-1)! \).

Step 4: Arrange the members within each entity.

Within each party entity, the members can be arranged in \( k! \) ways, where \( k \) is the number of members in that party.

Step 5: Combine the arrangements.

Multiply the number of arrangements of the entities by the number of arrangements within each entity to get the total number of arrangements.

Step 6: Calculate the total number of arrangements.

Let's denote the total number of arrangements as \( T \).

\[ T = (3-1)! \\times (5!) \\times (5!) \\times (1!) \]

\[ T = 2! \\times 120 \\times 120 \\times 1 \]

\[ T = 2 \\times 120 \\times 120 \]

\[ T = 28800 \]

Step 7: # Answer

28800
</attempt>
'''},
    {"role": "assistant", "content": '''Step 1: Identify the number of Democrats, Republicans, and Independents.

The identification of the number of members in each party is correct: 5 Democrats, 5 Republicans, and 1 Independent.

Conclusion: Step 1 is correct.

Step 2: Arrange the members of each party together.

This step correctly treats each group as a single entity due to the requirement that members of the same party sit next to each other.

Conclusion: Step 2 is correct.

Step 3: Arrange the entities around the circular table.

The logic here is to treat the three groups (Democrats, Republicans, Independent) as entities that are arranged around a circular table. Correctly, the number of ways to arrange \(n\) distinct objects around a circle is \((n-1)!\). Here, \(n = 3\) (Democrats, Republicans, Independent), so the arrangements are \((3-1)! = 2!\).

Conclusion: Step 3 is correct.

Step 4: Arrange the members within each entity.

This step correctly states that members within each group can be arranged in \(k!\) ways, where \(k\) is the number of members in the group. Therefore, the Democrats and Republicans can each be arranged in \(5!\) ways, and the Independent in \(1!\) way.

Conclusion: Step 4 is correct.

Step 5: Combine the arrangements.

This step correctly multiplies the arrangements of the groups around the table by the internal arrangements within each group to find the total number of unique seating arrangements.

Conclusion: Step 5 is correct.

Step 6: Calculate the total number of arrangements.

This step calculates \( T = 2! \\times 5! \\times 5! \\times 1! = 2 \\times 120 \\times 120 \\times 1 = 28800 \).

Conclusion: Step 6 is correct.

Step 7: Answer

The final answer given is 28800, which follows correctly from the calculations and assumptions made in the previous steps.

Conclusion: Step 7 is correct.
'''},
    {"role": "user", "content": '''How do you evaluate the following incorrect attempt?

<problem>
A strictly increasing sequence of positive integers $a_1$, $a_2$, $a_3$, $\\dots$ has the property that for every positive integer $k$, the subsequence $a_{2k-1}$, $a_{2k}$, $a_{2k+1}$ is geometric and the subsequence $a_{2k}$, $a_{2k+1}$, $a_{2k+2}$ is arithmetic. Suppose that $a_{13} = 2016$. Find $a_1$.
</problem>

<attempt>
Step 1: Identify the pattern in the given sequence.

We have two types of subsequences:
1. Geometric subsequences: \(a_{2k-1}\), \(a_{2k}\), \(a_{2k+1}\) are in geometric progression.
2. Arithmetic subsequences: \(a_{2k}\), \(a_{2k+1}\), \(a_{2k+2}\) are in arithmetic progression.

We're given that \(a_{13} = 2016\), and we need to find \(a_1\). We'll try to establish relationships between terms in the sequence to solve for \(a_1\).

Step 2: Use the properties of geometric and arithmetic progressions to relate terms.

For a geometric progression, the ratio between consecutive terms is constant. Let's denote this common ratio as \(r\).

For an arithmetic progression, the difference between consecutive terms is constant. Let's denote this common difference as \(d\).

Step 3: Relate terms to find \(r\) and \(d\).

We know that \(a_{13} = 2016\), so \(a_{12}\), \(a_{13}\), \(a_{14}\) form a geometric progression. Similarly, \(a_{12}\), \(a_{13}\), \(a_{14}\) form an arithmetic progression.

For the geometric progression:
\[a_{13} = a_{12} \\times r\]
\[2016 = a_{12} \\times r\] ...(1)

For the arithmetic progression:
\[a_{14} - a_{13} = a_{13} - a_{12}\]
\[a_{14} - 2016 = 2016 - a_{12}\]
\[a_{14} + a_{12} = 4032\] ...(2)

We now have two equations involving \(a_{12}\) and \(r\).

Step 4: Use these relationships to solve for \(a_{12}\) and \(r\).

From equations (1) and (2):
\[a_{12} \\times r = 2016\]
\[a_{12} + a_{14} = 4032\]

We can solve this system of equations to find \(a_{12}\) and \(r\).

Step 4 (continued): Solve the system of equations to find \(a_{12}\) and \(r\).

From equations (1) and (2), we have:
\[a_{12} \\times r = 2016\] ...(1)
\[a_{12} + a_{14} = 4032\] ...(2)

We can rewrite equation (1) as:
\[a_{12} = \\frac{2016}{r}\]

Substituting this expression for \(a_{12}\) into equation (2), we get:
\[\\frac{2016}{r} + a_{14} = 4032\]

Now, we need to find \(a_{14}\) in terms of \(r\).

Step 4 (continued): Express \(a_{14}\) in terms of \(r\).

Given that \(a_{13}\), \(a_{14}\), \(a_{15}\) form a geometric progression, we have:
\[a_{14} = 2016 \\times r\]

Now, substitute this expression for \(a_{14}\) into the equation:
\[\\frac{2016}{r} + 2016 \\times r = 4032\]

Now, we solve for \(r\):
\[\\frac{2016}{r} + 2016r = 4032\]

Multiply both sides by \(r\) to clear the fraction:
\[2016 + 2016r^2 = 4032r\]

Rearrange the terms:
\[2016r^2 - 4032r + 2016 = 0\]

Divide both sides by 2016 to simplify:
\[r^2 - 2r + 1 = 0\]

This quadratic equation factors as:
\[(r - 1)^2 = 0\]

So, \(r = 1\).

Step 5: Find \(a_{12}\) and \(a_1\).

Using \(r = 1\) in equation (1):
\[a_{12} = \\frac{2016}{1} = 2016\]

Now, we can find \(a_1\). Since \(a_{12} = a_1 \\times r^{11}\), and \(r = 1\), we have:
\[a_1 = \\frac{a_{12}}{r^{11}} = \\frac{2016}{1^{11}} = 2016\]

So, \(a_1 = 2016\).

Step 6: # Answer

2016
</attempt>
'''},
    {"role": "assistant", "content": '''Step 1: Identify the pattern in the given sequence
The problem accurately identifies that subsequence \(a_{2k-1}\), \(a_{2k}\), \(a_{2k+1}\) forms a geometric progression and that subsequence \(a_{2k}\), \(a_{2k+1}\), \(a_{2k+2}\) forms an arithmetic progression. 
Conclusion: Step 1 is correct.

Step 2: Use the properties of geometric and arithmetic progressions to relate terms
The attempt correctly introduces \(r\) as the common ratio for the geometric subsequence and \(d\) as the common difference for the arithmetic subsequence. 
Conclusion: Step 2 is correct.

Step 3: Relate terms to find \(r\) and \(d\)
The attempt makes the mistake of assuming \(a_{12}\), \(a_{13}\), and \(a_{14}\) form both a geometric and an arithmetic progression. The subsequences mentioned in the problem are those of the form \(a_{2k-1}\), \(a_{2k}\), \(a_{2k+1}\) (geometric) and \(a_{2k}\), \(a_{2k+1}\), \(a_{2k+2}\) (arithmetic). Thus, it's incorrect to analyze \(a_{12}\), \(a_{13}\), and \(a_{14}\) as if they belong to both progressions simultaneously. 
Conclusion: Step 3 is incorrect.

<correction>
Step 3 (Corrected): Identify relevant subsequences
Instead of using \(a_{12}\), \(a_{13}\), and \(a_{14}\), focus on subsequences specifically described in the problem:
1. Geometric progression: \(a_{11}\), \(a_{12}\), \(a_{13}\) because \(2k-1 = 11\) when \(k = 6\).
2. Arithmetic progression: \(a_{12}\), \(a_{13}\), \(a_{14}\).

Step 4: Use equations to find \(a_{11}\), \(a_{12}\), and \(a_{13}\)
From the geometric progression \(a_{11}\), \(a_{12}\), and \(a_{13}\):
\[ a_{12} = a_{11} \cdot r, \quad a_{13} = a_{11} \cdot r^2 \]
Given \(a_{13} = 2016\), it follows that:
\[ a_{11} \cdot r^2 = 2016. \]

For the arithmetic progression \(a_{12}\), \(a_{13}\), and \(a_{14}\):
\[ a_{13} - a_{12} = a_{14} - a_{13} = d \]
Thus, we express \(a_{14}\) in terms of \(a_{13}\) and the difference \(d\):
\[ a_{14} = 2a_{13} - a_{12}. \]

By substituting \(a_{12} = a_{11} \cdot r\) and \(a_{13} = a_{11} \cdot r^2\) into the equation above:
\[ a_{14} = 2a_{11} \cdot r^2 - a_{11} \cdot r. \]

Step 5: Solve equations
From \( a_{11} \cdot r^2 = 2016 \) (as obtained earlier), we solve for \(a_{11}\) by assuming integer values of \(r\). Testing successive values for \(r\):
\[ r = 2 \]
\[ a_{11} \cdot 4 = 2016 \Rightarrow a_{11} = \\frac{2016}{4} = 504. \]

Conclusion: The value of \( a_1 \) is \( \\boxed{504} \).
</correction>
'''},
]

messages = messages[:1+args.num_shot*2]

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


with open(args.src, 'r') as f:
    instance_list = f.readlines()[:args.max_instance]
instance_list = [json.loads(instance) for instance in instance_list]

cnt_previous = 0
try:
    with open(args.tgt, 'r') as f:
        cnt_previous = len(f.readlines())
except:
    pass

def infer_instance(instance):
    
    pred_labels = []
    gold_labels = []

    new_instance = instance['question']

    problem = instance['question']['problem']
    ground_truth_solution = instance['question']['ground_truth_solution']
    new_instance['correct'] = True if instance['label']['finish_reason'] == 'solution' else False

    steps_completion_with_label = instance['label']['steps']
    steps_list = []
    
    for i, step in enumerate(steps_completion_with_label):
        rating = step['completions'][0]['rating']
        if rating == 0:
            rating = 1
        gold_labels.append(rating)
        if rating == "-1":
            break

    hint_sent = "Hint: All the steps are correct, and the attempt reached a correct answer." if new_instance['correct'] else f"Hint: There could be a mistake."

    for i, step in enumerate(instance['question']['pre_generated_steps']):
        steps_list.append(f"Step {i+1}: {step}")
    len_steps = len(steps_list)

    steps_str = '\n\n'.join(steps_list)

    prompt = f'''How do you evaluate the following attempt with respect to the problem?\n{hint_sent}\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - Please ensure that the output text does not include phrases implying the use of a hint, even though these resources are being utilized.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Conclusion: Step [i] is correct" or "Conclusion: Step [i] is incorrect". Words like "partially correct" are prohibited. \n - You shall not evaluate multiple steps at a time, so words like "Step 7 to Step 24:" or "Step 4 through 6" are forbidden. \n''' if new_instance['correct'] else \
    f'''How do you evaluate the following incorrect attempt with respect to the problem, with the help of reference solution? \n{hint_sent}\n\n<problem>\n{problem}\n</problem>\n\n<reference_solution>\n{ground_truth_solution}\n</reference_solution>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - Please ensure that the output text does not include phrases implying the use of a reference solution or hint, even though these resources are being utilized.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Conclusion: Step [i] is correct" or "Conclusion: Step [i] is incorrect". Words like "partially correct" are prohibited. \n - You shall not evaluate multiple steps at a time, so words like "Step 7 to Step 24:" or "Step 4 through 6" are forbidden. \n - Once a mistake is identified and stated, stop the evaluation, and enumerate the corrected steps starting from the step where the mistake was detected, and label this part of your response with <correction> at the start and </correction> at the end.''' + ''' Also, the final answer should be a single number, in the form \\boxed{}, at the final step.'''


    print('\n', prompt)
    new_messages = messages + [{"role": "user", "content": prompt}]

    all_pred_labels = []
    all_critics = []
    bad_critics = []
    bad_pred_labels = []
    all_refinements = []
    all_refine_answers = []
    all_refine_correct = []

    for sample_cnt in range(args.sample):
        output = infer_stream(new_messages, model=args.model, temperature=args.temperature, stop=args.stop, max_tokens=args.max_tokens, top_p=args.top_p)
        pred_labels = parse_label(output, len_steps)

        print(output)
                
        print(pred_labels)
        if new_instance['correct']:
            if -1 not in pred_labels and 0 not in pred_labels:  
                print("Correct process, correct answer.")
                all_critics.append(output)
                all_pred_labels.append(pred_labels)
                all_refinements.append('')
                all_refine_answers.append('')
                all_refine_correct.append(None)
                break
            else:
                bad_critics.append(output)
                bad_pred_labels.append(pred_labels)

        if not new_instance['correct']:
            # Not found the wrong step
            if -1 not in pred_labels:
                bad_critics.append(output)
                bad_pred_labels.append(pred_labels)
                continue

            refine_solution, refine_answer = parse_correction_and_changed_answer(output)
            
            all_refine_answers.append(refine_answer)
            correct = grade_answer(instance['question']['ground_truth_answer'], refine_answer)
            all_refine_correct.append(correct)

            # print(refine_solution, refine_answer, correct)

            # refine successfully
            if correct:
                all_critics.append(output)
                all_pred_labels.append(pred_labels)
                all_refinements.append(refine_solution)
                print("Wrong process, success correction.")
                break
            else:
                bad_critics.append(output)
                bad_pred_labels.append(pred_labels)

    new_instance["ground_truth_answer"] = instance['question']['ground_truth_answer']
    new_instance["pred_labels"] = all_pred_labels
    new_instance["gold_pred_labels"] = gold_labels
    new_instance["critics"] = all_critics
    new_instance["bad_critics"] = bad_critics
    new_instance["bad_pred_labels"] = bad_pred_labels
    new_instance["refinements"] = all_refinements
    new_instance["refine_answers"] = all_refine_answers
    new_instance["refine_correct"] = all_refine_correct

    return new_instance


with open(args.tgt, 'a') as fw:
    with Pool(args.batch_size) as pool:
        for new_instance in tqdm(pool.imap(infer_instance, instance_list[cnt_previous:]), total=len(instance_list[cnt_previous:])):
            print(json.dumps(new_instance), file=fw)
            fw.flush()
