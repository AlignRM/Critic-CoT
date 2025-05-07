import ujson as json
import re
import random


critic_data = []

file_path = "xxx"

with open(file_path, 'r') as f:
    critic_data += f.readlines()


critic_data = [json.loads(d) for d in critic_data]

# filter out instances that "critis" == []
print(len(critic_data))
critic_data = [d for d in critic_data if d['critics'] != [] and len(d['pred_labels']) > 0 and len(d['pred_labels'][0]) > 0 and 0 != d['pred_labels'][0][0]]
len_critic_data = len(critic_data)
print(len_critic_data)


new_data = []

len_critic_data = 0
len_correct_data = 0
len_incorrect_data = 0
for idx, d in enumerate(critic_data):
    problem = d['problem']
    if d['pre_generated_steps'] != [] and type(d['pre_generated_steps'][0]) == str:
        d['pre_generated_steps'] = [d['pre_generated_steps']]
    if d['refinements'] == []:
        d['refinements'] = [""]
    if type(d['correct']) != list:
        d['correct'] = [d['correct']]
    if 'pre_generated_answer' in d:
        d['predict_answer'] = [d['pre_generated_answer']]
    if 'ground_truth_answer' in d:
        d['answer'] = d['ground_truth_answer']
    if type(d['predict_answer']) != list:
        d['predict_answer'] = [d['predict_answer']]
    if type(d['refine_answers']) != list:
        d['refine_answers'] = [d['refine_answers']]
    
    ci = -1
    for pre_generated_steps, correct, critic, refinement, predict_answer, refine_answer in zip(d['pre_generated_steps'], d['correct'], d['critics'], d['refinements'], d['predict_answer'], d['refine_answers']):
        ci += 1
        steps_list = []
        for i, step in enumerate(pre_generated_steps):
            steps_list.append(f"Step {i+1}: {step}")
        len_steps = len(steps_list)

        steps_str = '\n\n'.join(steps_list)

        if refinement == "":
            refinement = None
        else:
            refinement = "<correction>" + refinement + "</correction>"
     
        if correct and refinement is None :
            # Correct solution, no need to refine.
            if critic.startswith("<correction>"):
                critic = critic[len("<correction>"):-len("</correction>")]
            critic = re.sub(r'<correction>.*</correction>', '', critic)
            instance = {
                "id": f"critic_{idx}_{ci}",
                "system_message": "You are a helpful assistant.",
                "conversations": [
                    {
                        "from": "human",
                        "value": f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Step [i] is correct" or "Step [i] is incorrect".'''
                    },
                    {
                        "from": "gpt",
                        "value": critic.strip()
                    }
                ]
            }
            new_data.append(instance)
            len_correct_data += 1
            len_critic_data += 1

        elif not correct and refinement is not None:
            # Wrong solution with refinement.

            # Extract step-wise critic
            try:
                step_wise_critc = re.findall(r".*Step \d+ is incorrect", critic, re.DOTALL)[0]
            except:
                # print(critic)
                continue
            
            # print(step_wise_critc)
            wrong_step_no = re.findall(r"Step (\d+) is incorrect", step_wise_critc)[0]
            correct_steps_str = '\n\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(pre_generated_steps[:int(wrong_step_no)-1])])
            critic_instance = {
                "id": f"critic_{idx}_{ci}",
                "system_message": "You are a helpful assistant.",
                "conversations": [
                    {
                        "from": "human",
                        "value": f'''How do you evaluate the following attempt with respect to the problem?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n\n**Notes**:\n - Please think step by step.\n - Your reasoning should precede any claims or conclusions you make to avoid unwarranted assertions.\n - At the end of the evaluation for each step, YOU MUST articulate the conclusion using the format "Step [i] is correct" or "Step [i] is incorrect".'''
                    },
                    {
                        "from": "gpt",
                        "value": step_wise_critc.strip()
                    }
                ]
            }
            

            try:
                wrong_step_criticism = re.findall(f"Step {wrong_step_no}" + r".*Step \d+ is incorrect.", critic, re.DOTALL)[0]
            except:
                continue
                # print(critic)
                # input()

            new_data.append(critic_instance)
            refinement_instance = {
                "id": f"refinement_{idx}_{ci}",
                "system_message": "You are a helpful assistant.",
                "conversations": [
                    {
                        "from": "human",
                        "value": f'''How do you refine the following attempt with respect to the problem, given the criticism?\n\n<problem>\n{problem}\n</problem>\n\n<attempt>\n{steps_str}\n</attempt>\n\n<criticism>\n{wrong_step_criticism}\n</criticism>'''
                    },
                    {
                        "from": "gpt",
                        "value": refinement.strip()
                    }
                ]
            }
            new_data.append(refinement_instance)
            len_critic_data += 1
            len_incorrect_data += 1
        
import random

random.shuffle(new_data)

output_file = "xxx"
print(output_file)
fw = open(output_file, "w")
print(len_critic_data)
json.dump(new_data, fw, indent=4)
print('len_correct_data:', len_correct_data)
print('len_incorrect_data:', len_incorrect_data)
print('ratio:', len_correct_data / len_incorrect_data)
