import json
import random
import shutil
from random import sample

import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

cache_dir = 'HUGGINGFACE_CACHE_DIR'

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

data = pd.read_csv("../data/test.csv", index_col=None)
if len(data) < 2000:
    data = pd.concat([data, pd.read_csv('../data/dev.csv', index_col=None)])

data = data.sample(frac=1, random_state=829).reset_index(drop=True)

total_performance = pd.DataFrame(columns=['model', 'accuracy', 'macro_f1'])

model_list = [
    "microsoft/phi-2",
    "meta-llama/Llama-2-7b-chat-hf",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
    "tiiuae/falcon-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "tiiuae/falcon-40b-instruct",
    "meta-llama/Llama-2-70b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "tiiuae/falcon-7b",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "tiiuae/falcon-40b",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-70B",
]


def parse_answer(generation):
    if 'no' in generation.lower().split():
        return 0
    elif 'yes' in generation.lower().split():
        return 1
    elif 'wrong' in generation.lower().split():
        return 0
    elif 'right' in generation.lower().split():
        return 1
    elif 'correct' in generation.lower().split():
        return 1
    elif 'incorrect' in generation.lower().split():
        return 0
    else:
        return random.choice([0, 1])


def print_performance(result):
    for column in result.columns:
        if column in model_list:
            accuracy = sum(result[column] == data['label']) / len(data)
            macro_f1 = f1_score(data['label'], result[column], average='macro')
            print("{} & {} & {} \\\\".format(column.split('/')[-1], round(accuracy, 4), round(macro_f1, 4)))


if not os.path.exists('./results/'):
    os.makedirs('./results/')

for model in model_list:
    tokenizer = AutoTokenizer.from_pretrained(model, token="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",
                                              cache_dir=cache_dir)
    loaded_model = AutoModelForCausalLM.from_pretrained(model,
                                                        cache_dir=cache_dir,
                                                        token="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",
                                                        device_map='auto')

    if loaded_model.num_parameters() > 30_000_000_000 and len(data) > 1500:
        data = data[:2000]

    zero_shot_generations, cot_generations = data.copy(), data.copy()

    count = 0

    for i in trange(len(data), desc="Generating zero-shot with {}".format(model)):
        prompt = """Given an assertion that describes a if-then inference, determine whether the inference is plausible or not.
        Assertion: {}. Answer Yes or No only. Your answer is""".format(
            "If {} then {}".format(data.loc[i, 'event_after_transition'],
                                   data.loc[i, 'inference']))

        zero_shot_sequences = loaded_model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids.to(loaded_model.device),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=len(tokenizer.encode(prompt)) + 3,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(zero_shot_sequences[0], skip_special_tokens=True)
        if count < 3:
            print(generated_text.split("Answer Yes or No only. Your answer is")[1])
            count += 1
        zero_shot_generations.loc[i, model] = parse_answer(
            generated_text.split("Answer Yes or No only. Your answer is")[1])

    zero_shot_generations.to_csv("./results/{}_zero_shot.csv".format(model.split('/')[-1]), index=False)

    print("\nZero-shot performance for model {}".format(model))
    print_performance(zero_shot_generations)
    print('\n')

    total_performance.loc[len(total_performance)] = [model.split('/')[-1],
                                                     sum(zero_shot_generations[model] == data['label']) / len(data),
                                                     f1_score(data['label'], zero_shot_generations[model],
                                                              average='macro')]

total_performance.to_csv("./results/total_performance.csv", index=False)
print(total_performance)
