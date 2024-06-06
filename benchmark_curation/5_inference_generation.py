import os.path
import sys
import warnings
from random import sample

import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from generate_with_openai import generate_with_openai

TEST_MODE = False

GENERATION_UPPER_LIMIT = 20000

CSV_PATH = [
    '../4_event_ensemble_and_sampling/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/ensemble.csv',
    '../4_event_ensemble_and_sampling/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/ensemble.csv',
    '../4_event_ensemble_and_sampling/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/ensemble.csv',
    '../4_event_ensemble_and_sampling/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/ensemble.csv',
]

prompt = """Given an event, first determine its type is whether an action or a state. 
    A state is a condition or situation in which someone or something exists in the past or present that will last for a certain time if no changes occur. 
    An action is a thing that can be done in a time interval that is usually not long.
    Then, generate a short if-then inferential statement that satisfies if event then inference. For example:
    Event: Sam drives down the road with fast speed.
    Type: Action
    Inference: Sam is in a hurry.

    Event: The sun is shining brightly.
    Type: State
    Inference: People can go out for a picnic.

    Event: In 2003, he had a recurring role on two episodes of The Bill.
    Type: State
    Inference: People enjoyed his performance.

    Event: The resident feels sad and lonely.
    Type: State
    Inference: The resident needs some company.

    Event: {}
    Type:"""


def inference_generation(event, verbose=False):
    prompt = """Given an event, first determine its type is whether an action or a state. 
    A state is a condition or situation in which someone or something exists in the past or present that will last for a certain time if no changes occur. 
    An action is a thing that can be done in a time interval that is usually not long.
    Then, generate a short if-then inferential statement that satisfies if event then inference. For example:
    Event: Sam drives down the road with fast speed.
    Type: Action
    Inference: Sam is in a hurry.
    
    Event: The sun is shining brightly.
    Type: State
    Inference: People can go out for a picnic.
    
    Event: In 2003, he had a recurring role on two episodes of The Bill.
    Type: State
    Inference: People enjoyed his performance.
    
    Event: The resident feels sad and lonely.
    Type: State
    Inference: The resident needs some company.
    
    Event: {}
    Type:"""
    success, result = generate_with_openai(prompt.format(event), verbose=verbose)
    try:
        generation = result[0].split('\n')
    except:
        return None, prompt.format(event)
    if len(generation) == 2 and success:
        return [generation[0], generation[1].replace('Inference:', '').strip()], result[1]
    else:
        return None, prompt.format(event)


def replace_all_sublist(event_tokens, instance_tokens, variation_tokens):
    # Find the length of the instance_tokens
    instance_len = len(instance_tokens)

    # Iterate over the event_tokens
    i = 0
    while i < len(event_tokens):
        # Check if the sublist starting from the current index matches instance_tokens
        if event_tokens[i:i + instance_len] == instance_tokens:
            # Replace the sublist with variation_tokens
            event_tokens[i:i + instance_len] = variation_tokens
            # Move the index forward by the length of variation_tokens
            i += len(variation_tokens)
        else:
            # Move to the next index
            i += 1

    # Join the event_tokens back into a string
    result = ' '.join(event_tokens)
    return result


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 5 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if 'length' in list(data):
        data = data.drop(columns=['length'])
    if TEST_MODE:
        data = data.sample(5).reset_index(drop=True)
    print("Number of event_discrimination_data to be generated:", len(data))
    if len(data['event_id'].unique()) > GENERATION_UPPER_LIMIT:
        sample_event_id = sample(data['event_id'].unique().tolist(), GENERATION_UPPER_LIMIT)
        data = data[data['event_id'].isin(sample_event_id)].reset_index(drop=True)
        unsampled_data = data[~data['event_id'].isin(sample_event_id)]
        unsampled_data.to_csv(
            './{}/{}/unsampled_data.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)

    # text_id, text, event_id, event, component_type, component_original, component_substitution
    data_failed_generation = pd.DataFrame(columns=['text_id', 'text', 'event_id', 'event', 'component_type',
                                                   'component_original', 'component_substitution', 'prompt'])
    total_price = 0
    progress_bar = tqdm(total=len(data), desc="Conceptualizing {}".format(CSV.split('/')[3].split('.')[0]))
    for i in range(len(data)):
        event = data['event'][i]
        instance = data['component_original'][i]
        variation = data['component_substitution'][i]
        event_tokens = event.lower().split()
        instance_tokens = instance.lower().split()
        variation_tokens = variation.lower().split()
        # print(event)
        # replace the instance in the event with the variation
        event = replace_all_sublist(event_tokens, instance_tokens, variation_tokens)
        # print(event)
        result, price = inference_generation(event, verbose=TEST_MODE)
        if result:
            data.loc[i, 'event_type'] = result[0]
            data.loc[i, 'inference'] = result[1]
            total_price += price
        else:
            data_failed_generation.loc[len(data_failed_generation)] = [data['text_id'][i], data['text'][i],
                                                                       data['event_id'][i], data['event'][i],
                                                                       data['component_type'][i],
                                                                       data['component_original'][i],
                                                                       data['component_substitution'][i],
                                                                       prompt.format(event)
                                                                       ]

        progress_bar.update(1)
        progress_bar.set_postfix(
            {"Total Price (HKD)": total_price, "Estimated:": total_price / (i + 1) * len(data)})

        if i % 1000 == 0:
            data.to_csv(
                './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
            data_failed_generation.to_csv(
                './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
                index=False)

    data.to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
