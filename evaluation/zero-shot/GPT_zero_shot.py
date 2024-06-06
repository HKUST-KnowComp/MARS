import os.path
import random
import sys

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import trange

sys.path.append('../../')
sys.path.append('../../../')
from generate_with_openai import generate_with_openai


def parse_answer(generation):
    if 'no' in generation.lower().split():
        return 0
    elif 'yes' in generation.lower().split():
        return 1
    elif 'wrong' in generation.lower().split():
        return 0
    elif 'right' in generation.lower().split():
        return 1
    elif 'incorrect' in generation.lower().split():
        return 0
    elif 'correct' in generation.lower().split():
        return 1
    else:
        return random.choice([0, 1])


def replace_substrings(string1, string2, string3):
    # Split string2 into words
    words = string2.split()

    # Split string1 and string3 into individual words
    words_to_replace = string1.split()

    if len(words_to_replace) > 1:
        return string2.replace(string1, string3)
    else:
        return " ".join([i if i != string1 else string3 for i in words])


log = open("log.txt", "w")

if not os.path.exists('./results'):
    os.mkdir('./results')

train_data = pd.read_csv("../data/train.csv", index_col=None)
data = pd.read_csv("../data/test.csv", index_col=None).sample(n=1000).reset_index(drop=True)

for model in ['gpt-35-turbo', 'gpt4']:
    for i in trange(len(data), desc="Inferring for model: {} (zero-shot)".format(model)):
        prompt = """
        Given an assertion with a if-then relationship, determine if it's plausible or not.
        If it's plausible and can possibly occur in reality world, answer yes.
        If not and it's a metaphysical inference, meaning that it's not commonly seen in the real world, answer no.
        Assertion: If {}, then {}
        Answer yes or no only.
        """.format(data.loc[i, 'event_after_transition'], data.loc[i, 'inference'])
        success, result = generate_with_openai(prompt, model=model, max_tokens=5, verbose=False)
        if success:
            data.loc[i, model + '|zs'] = parse_answer(result[0])
        else:
            data.loc[i, model + '|zs'] = random.choice([0, 1])

    data.to_csv("./results/{}-zero-shot.csv".format(model), index=False)

    # calculate average accuracy between answer and gpt4
    accuracy = sum(data['label'] == data[model + '|zs']) / len(data)
    macro_f1 = f1_score(data['label'], data[model + '|zs'], average='macro')
    print("Model: {} (zero-shot); Accuracy: {}; Macro-F1: {}".format(model, accuracy, macro_f1))
    log.write("{} (zero-shot) & {} & - & {}\n".format(model, accuracy, macro_f1))

    for i in trange(len(data), desc="Inferring for model: {} (5-shots)".format(model)):
        train_data_sample_5 = train_data.sample(n=5).reset_index(drop=True)
        prompt = """
        Given an assertion with a if-then relationship, determine if it's plausible or not.
        If it's plausible and can possibly occur in reality world, answer yes.
        If not and it's a metaphysical inference, meaning that it's not commonly seen in the real world, answer no.
        Answer yes or no only.
        Here are some examples:
        Assertion: If {}, then {}. Answer: {}
        Assertion: If {}, then {}. Answer: {}
        Assertion: If {}, then {}. Answer: {}
        Assertion: If {}, then {}. Answer: {}
        Assertion: If {}, then {}. Answer: {}
        Assertion: If {}, then {}
        Answer yes or no only.
        """.format(
            train_data_sample_5.loc[0, 'event_after_transition'], train_data_sample_5.loc[0, 'inference'],
            train_data_sample_5.loc[0, 'label'],
            train_data_sample_5.loc[1, 'event_after_transition'], train_data_sample_5.loc[1, 'inference'],
            train_data_sample_5.loc[1, 'label'],
            train_data_sample_5.loc[2, 'event_after_transition'], train_data_sample_5.loc[2, 'inference'],
            train_data_sample_5.loc[2, 'label'],
            train_data_sample_5.loc[3, 'event_after_transition'], train_data_sample_5.loc[3, 'inference'],
            train_data_sample_5.loc[3, 'label'],
            train_data_sample_5.loc[4, 'event_after_transition'], train_data_sample_5.loc[4, 'inference'],
            train_data_sample_5.loc[4, 'label'],
            data.loc[i, 'event_after_transition'], data.loc[i, 'inference']
        )

        success, result = generate_with_openai(prompt, model=model, max_tokens=10)
        if success:
            data.loc[i, model + '|5s'] = parse_answer(result[0])
        else:
            data.loc[i, model + '|5s'] = random.choice([0, 1])

    data.to_csv("./results/{}-5-shots.csv".format(model), index=False)

    # calculate average accuracy between answer and gpt4
    accuracy = sum(data['label'] == data[model + '|5s']) / len(data)
    macro_f1 = f1_score(data['label'], data[model + '|5s'], average='macro')
    print("Model: {} (5-shots); Accuracy: {}; Macro-F1: {}".format(model, accuracy, macro_f1))
    log.write("{} (5-shots) & {} & - & {}\n".format(model, accuracy, macro_f1))

    for i in trange(len(data), desc="Inferring for model: {} (chain-of-thought)".format(model)):
        prompt = """
        Given an assertion with a if-then relationship, determine if it's plausible or not.
        If it's plausible and can possibly occur in reality world, answer yes.
        If not and it's a metaphysical inference, meaning that it's not commonly seen in the real world, answer no.
        Assertion: If {}, then {}
        Think step by step and explain your answer with a short rationale first.
        Then, finally, answer yes or no only.
        """.format(data.loc[i, 'event_after_transition'], data.loc[i, 'inference'])
        success, result = generate_with_openai(prompt, model=model, max_tokens=10)
        if success:
            data.loc[i, model + '|cot'] = parse_answer(result[0])
        else:
            data.loc[i, model + '|cot'] = random.choice([0, 1])

    data.to_csv("./results/{}-cot.csv".format(model), index=False)

    # calculate average accuracy between answer and gpt4
    accuracy = sum(data['label'] == data[model + '|cot']) / len(data)
    macro_f1 = f1_score(data['label'], data[model + '|cot'], average='macro')
    print("Model: {} (chain-of-thought); Accuracy: {}; Macro-F1: {}".format(model, accuracy, macro_f1))
    log.write("{} (chain-of-thought) & {} & - & {}\n".format(model, accuracy, macro_f1))

log.close()
