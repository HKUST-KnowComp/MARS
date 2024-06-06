import os.path
import sys
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
from generate_with_openai import generate_with_openai

TEST_MODE = False

CSV_PATH = [
    '../5_inference_generation/wikitext_length30-200_filter/wikitext_validation_sample100_length30-200/generation.csv',
    '../5_inference_generation/wikitext_length30-200_filter/wikitext_test_sample100_length30-200/generation.csv',
    '../5_inference_generation/wikitext_length30-200_filter/wikitext_train_sample50_length30-200/generation.csv',
    '../5_inference_generation/bookcorpus_length30-200_filter/bookcorpus_train_sample10_length30-200/generation.csv',
]

prompt = """Given an event and its event type as either an action or a state, generate a short metaphysical/counterfactual if-then inferential statement that describes an inferential knowledge that only occurs in metaphysical space. 
    A state is a condition or situation in which someone or something exists in the past or present that will last for a certain time if no changes occur.
    An action is a thing that can be done in a time interval that is usually not long.
    If the event is an action, generate a state in the inference. If it's a state, generate an action in the inference.
    Metaphysical inference is a type of inference that is not based on empirical evidence but rather on the nature of things.
    It can be a counterfactual inference that is contrary to the facts or reality, meaning that it is usually not true in reality world. For example:
    Event: Sam drives down the road with fast speed.
    Type: Action
    Metaphysical Inference: Sam is not in a hurry and is enjoying the ride.

    Event: The sun is shining brightly.
    Type: State
    Metaphysical Inference: People are not going out for a picnic because it's raining.

    Event: In 2003, he had a recurring role on two episodes of The Bill.
    Type: State
    Metaphysical Inference: People criticize his performance in the show.

    Event: The resident is being chased by a 100 meters butterfly.
    Type: State
    Metaphysical Inference: He is not scared and is enjoying the chase.

    Event: {}
    Type: {}
    Metaphysical Inference:"""


def inference_generation(event, type, verbose=False):
    success, result = generate_with_openai(prompt.format(event, type), verbose=verbose)
    try:
        generation = result[0].split('Metaphysical Inference:')[-1].strip()
    except:
        return None, prompt.format(event, type)
    if generation and success:
        return generation, result[1]
    else:
        return None, prompt.format(event, type)


def replace_substrings(string1, string2, string3):
    # Split string2 into words
    words = string2.split()

    # Split string1 and string3 into individual words
    words_to_replace = string1.split()
    replacement_words = string3.split()

    if len(words_to_replace) > 1:
        return string2.replace(string1, string3)
    else:
        return " ".join([i if i != string1 else string3 for i in words])
    # # Iterate over the words and replace if they match string1
    # for i in range(len(words)):
    #     # Check if the current word is an exact match for string1
    #     if words[i] == string1:
    #         words[i] = string3
    #     else:
    #         # Check if string1 is a substring of the current word, preserving word boundaries
    #         start_index = words[i].find(words_to_replace[0])
    #         while start_index != -1:
    #             end_index = start_index + len(words_to_replace[0])
    #             if (
    #                     (start_index == 0 or not words[i][start_index - 1].isalnum())
    #                     and (end_index == len(words[i]) or not words[i][end_index].isalnum())
    #             ):
    #                 words[i] = words[i][:start_index] + string3 + words[i][end_index:]
    #             start_index = words[i].find(words_to_replace[0], start_index + 1)
    #
    # # Join the modified words back into a string
    # modified_string = ' '.join(words)
    # return modified_string


if TEST_MODE:
    warnings.warn("Test mode is enabled. Only the first 5 rows will be processed.")
for CSV in CSV_PATH:
    if not os.path.exists('./{}/'.format(CSV.split('/')[2])):
        os.makedirs('./{}/'.format(CSV.split('/')[2]))
    if not os.path.exists('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0])):
        os.makedirs('./{}/{}'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]))

    data = pd.read_csv(CSV, index_col=None)
    if TEST_MODE:
        data = data.sample(5).reset_index(drop=True)
    print("Number of metaphysical inferences to be generated:", len(data))

    # text_id, text, event_id, event, component_type, component_original, component_substitution
    data_failed_generation = pd.DataFrame(columns=['text_id', 'text', 'event_id', 'event', 'component_type',
                                                   'component_original', 'component_substitution', 'event_type',
                                                   'inference', 'prompt'])
    total_price = 0
    progress_bar = tqdm(total=len(data), desc="Generating {}".format(CSV.split('/')[3].split('.')[0]))
    for i in range(len(data)):
        event = data['event'][i]
        instance = data['component_original'][i]
        variation = data['component_substitution'][i]
        event_tokens = event.lower()
        instance_tokens = instance.lower()
        variation_tokens = variation.lower()
        # print(event)
        # replace the instance in the event with the variation
        event = replace_substrings(instance_tokens, event_tokens, variation_tokens)
        # print(event)
        result, price = inference_generation(event, data['event_type'][i], verbose=TEST_MODE)
        if result:
            data.loc[i, 'metaphysical_inference'] = result
            total_price += price
        else:
            data_failed_generation.loc[len(data_failed_generation)] = [
                data['text_id'][i], data['text'][i],
                data['event_id'][i], data['event'][i],
                data['component_type'][i],
                data['component_original'][i],
                data['component_substitution'][i],
                data['event_type'][i],
                data['inference'][i],
                prompt.format(event, data['event_type'][i])
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

    data.dropna().to_csv(
        './{}/{}/generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), index=False)
    data_failed_generation.to_csv(
        './{}/{}/failed_generation.csv'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]),
        index=False)
    print("Total Price: {}".format(total_price))
    open('{}/{}/total_price.txt'.format(CSV.split('/')[2], CSV.split('/')[3].split('.')[0]), 'w').write(
        "Total price is:" + str(total_price))
