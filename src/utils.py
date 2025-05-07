import openai
import os
from time import sleep
import re
import random
random.seed()


from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexGroupNode, LatexMacroNode


    
def extract_boxed_expressions_custom(text):
    expressions = []
    stack = []
    current_expr = ""
    i = 0
    while i < len(text):
        if text[i:i+7] == r"\boxed{":
            if stack:
                current_expr += text[i]
            stack.append("{")
            i += 7
        elif text[i] == "{" and stack:
            stack.append("{")
            current_expr += text[i]
            i += 1
        elif text[i] == "}" and stack:
            stack.pop()
            if stack:
                current_expr += text[i]
            else:
                current_expr = re.split('=', current_expr)[-1]
                return current_expr
                # expressions.append(current_expr)
                current_expr = ""
            i += 1
        elif stack:
            current_expr += text[i]
            i += 1
        else:
            i += 1
    return ''


def parse_label(output, len_steps):
    pred_labels = []
    if output is None:
        return pred_labels
        
    for i in range(len_steps):
        label = 0
        if f"Step {i+1} is correct" in output:
            label = 1
        elif f"Step {i+1} is incorrect" in output:
            label = -1
        # else:
            # print("Wrong format.")
        pred_labels.append(label)
        if label == -1:
            break
    return pred_labels


def parse_correction_and_changed_answer(output, final_label=False):
    predict_solution, predict_answer = '', ''
    if final_label:
        try:
            # extract answer only, directly return the whole output
            predict_solution = output
            predict_answer_latex_str = re.findall(r'(\\boxed\{.*\})', predict_solution, re.DOTALL)[-1]
            predict_answer = extract_boxed_expressions_custom(predict_answer_latex_str)
        except Exception as e:
            # print(e)
            pass

        return predict_solution, predict_answer    
    try:
        # multiple line
        predict_solution = re.findall(r"<correction>(.*)</correction>", output, re.DOTALL)[0]
        predict_solution = re.sub(r'</correction>.*', '', predict_solution)
        predict_answer_latex_str = re.findall(r'(\\boxed\{.*\})', predict_solution, re.DOTALL)[-1]
        predict_answer = extract_boxed_expressions_custom(predict_answer_latex_str)
    except Exception as e:
        # print(e)
        pass

    return predict_solution, predict_answer
