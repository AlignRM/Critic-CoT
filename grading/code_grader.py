import contextlib
import io
import itertools
import json
import multiprocessing
import os.path as osp
import re
import signal
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import environ
from typing import List, Sequence, Union

import numpy as np


class TimeOutException(Exception):
    pass


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


def grade_answer(predictions, references):
    if len(predictions) != len(references):
        return {'error': 'preds and refrs have different length'}

    # result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
    details = {}
    num_processes = 32
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i, (refer, pred) in enumerate(zip(references,
                                                predictions)):
            pred = _process_answer(pred)
            programs = _process_test(refer, pred)
            future = executor.submit(execution, programs, i, 10)
            futures.append(future)
            details[str(i)] = {}
            details[str(i)]['origin'] = predictions[i]
            details[str(i)]['programs'] = pred

        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            index, ret = future.result()
            # result[ret] += 1
            details[str(index)]['result'] = ret
            details[str(index)]['is_correct'] = (ret == 'pass')

    # result['score'] = result['pass'] / len(predictions) * 100
    # result['details'] = details
    # correct_list = [details[str(i)]['is_correct'] for i in range(len(predictions))]
    # program_list = [details[str(i)]['programs'].split('\n') for i in range(len(predictions))]
    correct_list, program_list = [], []
    # Remove duplicate program
    program_set = set()
    for i in range(len(predictions)):
        program = details[str(i)]['programs']
        correct = details[str(i)]['is_correct']
        if program not in program_set:
            program_list.append(program.split('\n'))
            correct_list.append(correct)
            program_set.add(program)

    return correct_list, program_list, details

def _process_answer(text):
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
        r'BEGIN\s*(.*)\s*\[DONE\]',
        r'\[BEGIN\]\s*(.*)\s*DONE',
        r'BEGIN\s*(.*)\s*DONE',
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```Python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
        r"\[BEGIN\]\s*'(.*)",
        r'\[BEGIN\](.*)',
        r"'(.*)'\s*\[DONE\]",
    ]
    for p in patterns:
        match = re.findall(p, text, re.DOTALL)
        if match != []:
            text = match[-1]
            break
    text_split = text.split('```')
    text = text_split[-1]
    if "def" not in text and len(text_split) > 2:
        text = text_split[-3]
        
        
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = re.sub(r"\nLine \d+:", "\n", text)
    text = text.replace('\\_', '_')
    text = text.strip()
    if text.startswith("python\n") or text.startswith("Python\n"):
        text = text[7:]
    if text.startswith("<correction>"):
        text = re.findall(r'<correction> *(.*)</correction>', text, re.DOTALL)[0]
    return text

def _process_test(test_case, pred):
    formatted = pred + '\n'
    formatted += test_case
    return formatted


def _process_answer_2(text):
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith(
                    '\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    else:
        match = re.search(r'Here(.*?)\n', text)
        if match:
            text = re.sub('Here(.*?)\n', '', text, count=1)

    # remove test in generation
    test_list = ['# Test', '#Test', '#test', '# test']
    for s in test_list:
        if s in text:
            text = text[:text.find(s)]

    text = text.strip()
    match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
    if match:
        text = text[:match.start()]
    match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
    if match:
        text = text[match.end():]
    text = text.strip()
    if text.startswith("'"):
        text = text[1:]
    return text


def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return task_id, 'timeout'
    return task_id, key[0]

if __name__ == '__main__':
    text = "Step 1: We need to define a function filter_integers that takes a list of any python values as an argument.\n\nStep 2: We need to create an empty list to store the integers from the given list.\n\nStep 3: We need to iterate over each element in the given list.\n\nStep 4: For each element, we need to check if it is an integer. If it is, we add it to the list of integers.\n\nStep 5: Finally, we return the list of integers.\n\nHere is the Python code that implements these steps:\n\n```python\nfrom typing import List, Any\n\ndef filter_integers(values: List[Any]) -> List[int]:\n    \"\"\" Filter given list of any python values only for integers \"\"\"\n    integers = []\n    for value in values:\n        if isinstance(value, int):\n            integers.append(value)\n    return integers\n```\n\nThis function works as expected according to the provided unit tests:\n\n```python\nprint(filter_integers(['a', 3.14, 5]))  # [5]\nprint(filter_integers([1, 2, 3, 'abc', {}, []]))  # [1, 2, 3]\n```"
    print(_process_answer(text))
    exit()
    predictions = [
        "import time\ndef test(x):\n    time.sleep(11)\n    return x + 1\n",
        "def test(x):\n    return x + 2\n",
        "def test(x):\n    return x + 3\n",
    ]
    references = [
        "import math\nassert math.isclose(test(1), 2)\nassert test(2) == 3\n",
        "import math\nassert math.isclose(test(1), 2)\nassert test(2) == 3\n",
        "import math\nassert math.isclose(test(1), 2)\nassert test(2) == 3\n",
    ]
    print(grade_answer(predictions, references))